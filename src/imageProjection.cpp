
/************************************************* 
--------------------------------------------------
功能简介:
    1、利用当前激光帧起止时刻间的imu数据计算旋转增量，IMU里程计数据（来自ImuPreintegration）计算平移增量，进而对该帧激光每一时刻的激光点进行运动畸变校正（利用相对于激光帧起始时刻的位姿增量，变换当前激光点到起始时刻激光点的坐标系下，实现校正）；
    2、同时用IMU数据的姿态角（RPY，roll、pitch、yaw）、IMU里程计数据的的位姿，对当前帧激光位姿进行粗略初始化。

订阅：
    1、订阅原始IMU数据；
    2、订阅IMU里程计数据，来自ImuPreintegration，表示每一时刻对应的位姿；
    3、订阅原始激光点云数据。

发布：
    1、发布当前帧激光运动畸变校正之后的有效点云，用于rviz展示；
    2、发布当前帧激光运动畸变校正之后的点云信息，包括点云数据、初始位姿、姿态角、有效点云数据等，发布给FeatureExtraction进行特征提取。
*************************************************/  


#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

/**
 * Velodyne点云结构，变量名XYZIRT是每个变量的首字母
*/
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D // 位置
    PCL_ADD_INTENSITY; // 激光点反射强度，也可以存点的索引
    uint16_t ring; // 扫描线
    float time; // 时间戳，记录相对于当前帧第一个激光点的时差，第一个点time=0
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16; // 内存16字节对齐，EIGEN SSE优化要求

// 注册为PCL点云格式
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)


/**
 * Ouster点云结构
*/
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
// 本程序使用Velodyne点云结构

using PointXYZIRT = VelodynePointXYZIRT;

// imu数据队列长度
const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    // imu队列、odom队列的互斥锁
    std::mutex imuLock;
    std::mutex odoLock;

    // 订阅原始激光点云
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;

    // 发布当前帧校正后点云，有效点
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;

    // imu数据队列（原始数据，转lidar系下）
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    std::deque<sensor_msgs::msg::Imu> imuQueue;

    // 里程计队列
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
    std::deque<nav_msgs::msg::Odometry> odomQueue;

    // 激光点云数据队列
    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;
    
    // 队列front帧，作为当前处理帧点云
    sensor_msgs::msg::PointCloud2 currentCloudMsg;

    // 当前激光帧起止时刻间对应的imu数据，计算相对于起始时刻的旋转增量，以及时时间戳；
    // 用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    // 当前帧原始激光点云
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    // 当期帧运动畸变校正之后的激光点云
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    // 从fullCloud中提取有效点
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int ringFlag = 0;
    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    // 当前激光帧起止时刻间对应的imu数据，计算相对于起始时刻的旋转增量，以及时时间戳；
    // 用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    // 当前帧激光点云运动畸变校正之后的数据，包括点云数据、初始位姿、姿态角等，
    // 发布给featureExtraction进行特征提取
    lio_sam::msg::CloudInfo cloudInfo;
    // 当前帧起始时刻
    double timeScanCur;
    // 当前帧结束时刻
    double timeScanEnd;
    // 当前帧header，包含时间戳信息
    std_msgs::msg::Header cloudHeader;

    vector<int> columnIdnCountVec;


public:
    /**
     * 构造函数
    */
    ImageProjection(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imageProjection", options), deskewFlag(0)
    {
        callbackGroupLidar = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto lidarOpt = rclcpp::SubscriptionOptions();
        lidarOpt.callback_group = callbackGroupLidar;
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        // 订阅原始imu数据
        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&ImageProjection::imuHandler, this, std::placeholders::_1),
            imuOpt);
        // 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿
        subOdom = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic + "_incremental", qos_imu,
            std::bind(&ImageProjection::odometryHandler, this, std::placeholders::_1),
            odomOpt);
        // 订阅原始lidar数据
        subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar,
            std::bind(&ImageProjection::cloudHandler, this, std::placeholders::_1),
            lidarOpt);


        // 发布当前激光帧运动畸变校正后的点云，有效点
        pubExtractedCloud = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/deskew/cloud_deskewed", 1);
        // 发布当前激光帧运动畸变校正后的点云信息
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos);
        
        // 初始化
        allocateMemory();
        // 重置参数
        resetParameters();
        // pcl日志级别，只打ERROR日志
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.start_ring_index.assign(N_SCAN, 0);
        cloudInfo.end_ring_index.assign(N_SCAN, 0);

        cloudInfo.point_col_ind.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.point_range.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }
    
    /**
     * 重置参数，接收每帧lidar数据都要重置这些参数
    */
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }
    //~ImageProjection()是一个空的析构函数，它表示当ImageProjection对象被销毁时，不需要执行任何操作。
    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {
        // imu原始测量数据转换到lidar系，加速度、角速度、RPY
        sensor_msgs::msg::Imu thisImu = imuConverter(*imuMsg);

        // 上锁，添加数据的时候队列不可用
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf2::Quaternion orientation;
        // tf2::fromMsg(thisImu.orientation, orientation);
        // tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }


    /**
     * 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿
    */
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    /**
     * 订阅原始lidar数据
     * 1、添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
     * 2、当前帧起止时刻对应的imu数据、imu里程计数据处理
     *   imu数据：
     *   1) 遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     *   2) 用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
     *   imu里程计数据：
     *   1) 遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
     *   2) 用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
     * 3、当前帧激光点云运动畸变校正
     *   1) 检查激光点距离、扫描线是否合规
     *   2) 激光运动畸变校正，保存激光点
     * 4、提取有效激光点，存extractedCloud
     * 5、发布当前帧校正后点云，有效点
     * 6、重置参数，接收每帧lidar数据都要重置这些参数
    */
    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
    {
        // 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 检查当前帧起止时刻对应的imu数据，如果没有IMU数据就不进行点云处理
        if (!deskewInfo())
            return;


        // 当前帧激光点云运动畸变校正
        // 1、检查激光点距离、扫描线是否合规
        // 2、激光运动畸变校正，保存激光点
        projectPointCloud();

        // 提取有效激光点，存extractedCloud
        cloudExtraction();
        
        // 发布当前帧校正后点云，有效点
        publishClouds();
        // 重置参数，接收每帧lidar数据都要重置这些参数
        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        
        // 取出激光点云队列中最早的一帧
        currentCloudMsg = std::move(cloudQueue.front()); 
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            // 转换成pcl点云格式
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);  
        }
        else if (sensor == SensorType::OUSTER)
        {
            // 转换成Velodyne格式
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
            rclcpp::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        // 当前帧起始时刻
        timeScanCur = stamp2Sec(cloudHeader.stamp);
        // 当前帧结束时刻，注：点云中激光点的time记录相对于当前帧第一个激光点的时差，第一个点time=0
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;
    
        // remove Nan 存在无效点Nan需消除
        vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
            rclcpp::shutdown();
        }

        // check ring channel 检查是否存在ring通道，注意static只检查一次
        // we will skip the ring check in case of velodyne - as we calculate the ring value downstream (line 572)
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                if (sensor == SensorType::VELODYNE) {
                    ringFlag = 2;
                } else {
                    RCLCPP_ERROR(get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
                    rclcpp::shutdown();
                }
            }
        }

        // check point time 
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                //检查是否存在time通道 field.name可能是t或time
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                RCLCPP_WARN(get_logger(), "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /**
     * imu数据、imu里程计数据处理结果是否成功
    */
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // 要求Imu队列有imu数据，无数据不进行其他处理
        if (imuQueue.empty() ||
            stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
            stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd)
        {
            RCLCPP_INFO(get_logger(), "Waiting for IMU data ...");
            return false;
        }
        // 当前帧对应imu数据处理
        imuDeskewInfo();
        // 当前帧对应imu里程计处理
        odomDeskewInfo();

        return true;
    }

    /**
     * 当前帧对应imu数据处理
     * 1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     * 2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
     * 注：imu数据都已经转换到lidar系下了
    */
    void imuDeskewInfo()
    {
        cloudInfo.imu_available = false;
        //取lidar帧起止时刻之间的imu数据
        while (!imuQueue.empty())
        {
            if (stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::msg::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = stamp2Sec(thisImuMsg.header.stamp);

            // 提取imu姿态角RPY，作为当前lidar帧初始姿态角
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity 获取角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation 旋转角度积分
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imu_available = true;
    }


    /** 当前帧对应imu里程计处理
     *  1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
     *  2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
     * 注：imu数据都已经转换到lidar系下了
     **/
    void odomDeskewInfo()
    {
        cloudInfo.odom_available = false;
        //取lidar帧起止时刻之间的odom数据
        while (!odomQueue.empty())
        {
            if (stamp2Sec(odomQueue.front().header.stamp) < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (stamp2Sec(odomQueue.front().header.stamp) > timeScanCur)
            return;

        // 获取lidar帧初始时刻的odom
        nav_msgs::msg::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (stamp2Sec(startOdomMsg.header.stamp) < timeScanCur)
                continue;
            else
                break;
        }

        tf2::Quaternion orientation;
        tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization 使用了初始假设
        cloudInfo.initial_guess_x = startOdomMsg.pose.pose.position.x;
        cloudInfo.initial_guess_y = startOdomMsg.pose.pose.position.y;
        cloudInfo.initial_guess_z = startOdomMsg.pose.pose.position.z;
        cloudInfo.initial_guess_roll = roll;
        cloudInfo.initial_guess_pitch = pitch;
        cloudInfo.initial_guess_yaw = yaw;

        cloudInfo.odom_available = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (stamp2Sec(odomQueue.back().header.stamp) < timeScanEnd)
            return;

        nav_msgs::msg::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (stamp2Sec(endOdomMsg.header.stamp) < timeScanEnd)
                continue;
            else
                break;
        }
        //起始和结束Odom数据匹配,协方差不同时，认为数据不匹配，直接返回， odomDeskewFlag = false
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }


    /* imuTime队列中每个点都有这个点的时间。
     * 按照给定点的时间，找到对应的imu旋转数据，赋值给当前点的旋转变量
    */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        // 取到imu时间队列中，符合给定的pointtime的第一个imuPointFront点 
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        //直接将imuPointerFront对应的数据作为给定pointTime时间点的初始姿态
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
          /* 给定的pointTime时间点的imu数据在imuTime队列中间的某一个点，imuPointerFront ！= 0
           * 需要按照点的时间进行插值计算，找到对应的imu旋转数据，赋值给当前点的旋转变量
           */
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;
        //参照照相机原理 ，原点移动对最终成像影响很小，因此非常小的时间差内移动直接忽略
        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        /* 如果畸形矫正标识为-1，或者缺失imu数据，说明不需要进行畸形矫正，直接返回原始数据
         * 因为IMU数据（例如姿态和加速度）通常用于精确补偿运动，故不进行去畸变。
         * 但没有IMU进行矫正，会产生严重的漂移
         */
        if (deskewFlag == -1 || cloudInfo.imu_available == false)
            return *point;

        /* Lidar数据中有这个reltime值, laserCloudIn->points[i].time
         *   relTime： 表示当前激光点相对于整个激光帧开始时间的相对时间。但findPosition实际实现中忽略的这个reltime
         *   pointTime： 是当前点的绝对时间戳，通过帧起始时间 timeScanCur 加上该点的相对时间 relTime 得到。
         */ 
        double pointTime = timeScanCur + relTime;

        /* 旋转增量：
         * 这一步根据点的时间戳 pointTime 计算该时刻的旋转增量，rotXCur、rotYCur 和 rotZCur 是相对于激光帧起始时刻的旋转增量（绕X、Y、Z轴的旋转角度）。
         */
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        /* 平移增量：
         * 这一步根据相对时间 relTime 计算该时刻的平移增量，posXCur、posYCur 和 posZCur 是相对于起始时刻的平移增量。
        */
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        /* 记录第一点的逆变换：
         * 如果是处理的第一个点，计算它的变换矩阵（包括旋转和平移），
         * 并且求取其逆矩阵 transStartInverse，用于将后续点相对于第一个点的运动进行去畸变。
         * firstPointFlag 标志在处理完第一个点后被置为 false，确保这个过程只在第一个点时发生。
        */
        if (firstPointFlag == true)
        {
            /* transStart是 Z(yaw)Y(pitch)X(roll)外旋，就是先转x，再转y，最后转z的变换矩阵 ，生成一个从世界坐标系到当前坐标系的变换矩阵。
             * 通过.inverse() 方法获得这个变换的逆矩阵 transStartInverse。
             * transStartInverse是逆矩阵 ，也就是说，这个矩阵将当前坐标系下的点转换回世界坐标系。
             */
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        /* 计算当前点的变换（transform points to start）
         * rotXCur、rotYCur 和 rotZCur 是相对于激光帧起始时刻的旋转增量
         * posXCur、posYCur 和 posZCur 是相对于起始时刻的平移增量。
         * transFinal 是根据当前点的旋转和平移增量，计算出来的当前时刻相对于激光帧起始时刻的变换矩阵。
        */
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);

        /* 当前时刻激光点与第一个激光点的位姿变换:
         * 通过 transStartInverse 和 transFinal 的矩阵相乘 transBt，得到从第一个点开始到当前点的相对变换。这个变换会用于去畸变。
         * transBt = transStartInverse * transFinal矩阵乘法表示的是：
         * 先将当前坐标系的点通过 transStartInverse 转换回世界坐标系，再通过 transFinal 转换回当前的坐标系。
        */
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        
        // 当前激光点在第一个激光点坐标系下的坐标
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }
    

    /**
     * 当前帧激光点云运动畸变校正
     * 1、检查激光点距离、扫描线是否合规
     * 2、激光运动畸变校正，保存激光点
    */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection 
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            
            //这里用半径作为过滤点云的条件，如果需要用其他条件过滤，在这里修改。
            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            // if sensor is a velodyne (ringFlag = 2) calculate rowIdn based on number of scans

            /*
             * 这里计算以-90 ~ 90为扫描线rowIdn的计算，只适合与激光雷达线数小于90的情况，
             * 中心点为激光线数/2，计算公式为：rowIdn = (verticalAngle + (N_SCAN - 1)) / 2，
             * 如果激光线分布不均匀 或者激光线数大于90，均要调整算法
              */ 
            if (ringFlag == 2) { 
                float verticalAngle =
                    atan2(thisPoint.z,
                        sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) *
                    180 / M_PI;
                rowIdn = (verticalAngle + (N_SCAN - 1)) / 2.0;
            }

            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;


            /*
              注意atan2(thisPoint.x, thisPoint.y)，X 是前方、y是右方，代表正前方的角度为0度
            */
            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            
            /* Livox 雷达的点云数据结构与 Velodyne 或 Ouster 等传统机械式旋转激光雷达不同，它采用了非旋转的扫描机制，使用了非重复采样模式，
             * 这意味着 Livox 雷达采集的点云并不会均匀地覆盖空间，而是随机分布在整个视野中。
             * 因此，传统的基于水平角度计算列索引的方式（如 Velodyne 使用 atan2 来计算水平角度）并不适用于 Livox 雷达。

             * 非重复采样模式：
                 Livox 雷达的扫描模式不是固定的旋转式，而是基于其特有的非重复采样技术。它在每次扫描中，采样的点在空间中的分布是随机的、非均匀的。这样做的好处是，随着时间的推移，雷达可以更均匀地覆盖整个视野，但在单次扫描中，点的分布并不规则。
               因此，在每个扫描线（rowIdn）上，点的水平分布是动态变化的，不能像 Velodyne 那样通过固定的角度关系来计算每个点在范围图中的列索引。
               
            * 随机采样导致无法直接通过角度计算列索引：
                 对于 Velodyne 或 Ouster 这样的旋转雷达，点云中的每个点都有明确的水平角度，可以通过水平角度直接计算出在二维范围图中的列索引。
              而Livox 的点没有固定的水平角度顺序，采集的点没有均匀的排列，因此无法通过 atan2 来计算每个点的列索引。
              
            * 在这种情况下，使用 columnIdnCountVec 来累加列索引：
                由于 Livox 雷达的点分布是随机的，代码中使用了 columnIdnCountVec 来对每个扫描线（rowIdn）的列索引进行累加。
                这意味着它不依赖于点的水平角度，而是按照点在该扫描线上的出现顺序分配列索引。
                每次在该扫描线上处理一个点时，就将该点分配到下一个可用的列索引，并将对应的 columnIdnCountVec[rowIdn] 累加1。
                这种方式能够确保所有点都能被有效地投影到二维范围图中，尽管它们的水平分布是不规则的。
            */
           else if (sensor == SensorType::LIVOX)
           {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }


            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            /* rangeMat 是一个 OpenCV 的 cv::Mat 类型的二维矩阵，通常用于存储点云数据的距离信息，形成所谓的“范围图（range image）”
              rangeMat.at<float>(rowIdn, columnIdn) 表示访问 rangeMat 矩阵中位于 (rowIdn, columnIdn) 的元素，
              该元素是一个 float 类型的浮点数，通常表示激光雷达点云的距离值（range）。
            */
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {   
            /* 表示跳过前 5 个点。这种处理通常用于避免扫描线两端的噪声点，
             * 因为激光雷达的每条扫描线开头和结尾的点容易受到不稳定因素的影响
             *（例如，边缘点由于视角原因可能不准确）。
             */
            cloudInfo.start_ring_index[i] = count - 1 + 5;
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.point_col_ind[count] = j;
                    // save range info
                    cloudInfo.point_range[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            //表示跳过扫描线末尾的 5 个点，原因与上面类似，是为了避免使用靠近扫描线末端的噪声点。
            cloudInfo.end_ring_index[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<ImageProjection>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Image Projection Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
