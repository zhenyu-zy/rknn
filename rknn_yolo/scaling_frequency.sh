# 初始化芯片名称变量，默认为 0
chip_name=0
# 初始化频率设置状态变量，0 表示设置成功，-1 表示设置失败
freq_set_status=0

# 定义 usage 函数，用于显示脚本的使用说明
usage()
{
    echo "USAGE: ./scaling_frequency.sh -c rk3588 [-h]"
    echo "  -c:  chip_name, only support rk3588"
    echo "  -h:  Help"
}

# 定义 print_and_compare_result 函数，用于打印尝试设置的频率和查询到的当前频率，并比较两者是否一致
# 参数 $1 为尝试设置的频率，参数 $2 为查询到的当前频率
print_and_compare_result()
{
    echo "    try set "$1
    echo "    query   "$2
    if [ "$1" == "$2" ];then
        echo "    Setting Success"
    else
        echo "    Setting Failed"
        freq_set_status=-1
    fi
}

# 定义 print_not_support_adjust 函数，用于打印固件不支持设置特定设备频率的信息
# 参数 $1 为设备名称，参数 $2 为想要设置的频率，参数 $3 为查询到的当前频率
print_not_support_adjust()
{
    echo "Firmware seems not support setting $1 frequency"
    echo "    wanted "$2
    echo "    check  "$3
}


# 主函数开始
# 初始化有效标志变量，用于判断输入是否有效
vaild=0

# 如果没有提供任何命令行参数，则显示使用说明并退出脚本
if [ $# == 0 ]; then
    usage
    exit 0
fi

# 使用 getopts 解析命令行参数
while getopts "c:h" arg
do
    case $arg in
        c)
          # 如果参数为 -c，则将其后的参数赋值给 chip_name 变量
          chip_name=$OPTARG
          # 仅支持rk3588
          if [ $chip_name != "rk3588" ]; then
              echo "Only rk3588 is supported"
              usage
              exit 1
          fi
          ;;
        h)
          # 如果参数为 -h，则显示使用说明并退出脚本
          usage
          exit 0
          ;;
        ?)
          # 如果参数无效，则显示使用说明并以错误状态码 1 退出脚本
          usage
          exit 1
          ;;
    esac
done

# 根据芯片名称设置不同的频率和设置策略
if [ $chip_name == 'rk3588' ]; then
    seting_strategy=4
    CPU_freq=2256000  # 所有CPU核心最高频率
    NPU_freq=1000000000  # NPU最高频率
    DDR_freq=2112000000  # DDR最高频率
else 
    # 如果芯片名称不被识别，则输出错误信息并以错误状态码 1 退出脚本
    echo "$chip_name not recognize, only rk3588 is supported"
    exit 1
fi

# 输出尝试设置频率的芯片名称、设置策略以及要设置的 CPU、NPU 和 DDR 频率
echo "Setting maximum frequency for "${chip_name}
echo "    Setting strategy as $seting_strategy"
echo "    NPU frequency: "$NPU_freq
echo "    CPU frequency: "$CPU_freq
echo "    DDR frequency: "$DDR_freq

# 根据设置策略执行不同的频率设置操作
case $seting_strategy in
    4)
        # 禁用 CPU 的空闲状态 1-7
        for i in $(seq 0 7); do
            echo 1 > /sys/devices/system/cpu/cpu$i/cpuidle/state1/disable
        done

        # 设置所有CPU核心频率
        echo "CPU: Setting frequency for all cores"
        for i in $(seq 0 7); do
            echo "  Core$i"
            # 将 CPU 核心的频率调节策略设置为 userspace
            echo userspace > /sys/devices/system/cpu/cpufreq/policy$i/scaling_governor
            # 设置 CPU 核心频率
            echo $CPU_freq > /sys/devices/system/cpu/cpufreq/policy$i/scaling_setspeed
            # 获取当前 CPU 核心频率
            current_freq=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_cur_freq)
            # 比较设置的 CPU 核心频率和当前频率
            print_and_compare_result $CPU_freq $current_freq
            # 存储最后一个核心的频率用于结果记录
            CPU_cur_freq=$current_freq
        done

        echo "NPU: Setting frequency"
        if [ -e  /sys/class/devfreq/fdab0000.npu/governor ];then
            # 如果存在该文件，将 NPU 频率调节策略设置为 userspace
            echo userspace > /sys/class/devfreq/fdab0000.npu/governor 
            # 设置 NPU 频率
            echo $NPU_freq > /sys/class/devfreq/fdab0000.npu/userspace/set_freq 
            # 获取当前 NPU 频率
            NPU_cur_freq=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq)
        elif [ -e /sys/class/devfreq/devfreq0/governor ];then
            # 如果存在该文件，将 NPU 频率调节策略设置为 performance
            echo performance > /sys/class/devfreq/devfreq0/governor 
            # 获取当前 NPU 频率
            NPU_cur_freq=$(cat /sys/class/devfreq/devfreq0/cur_freq)
        else
            # 从内核调试信息中获取 NPU 频率
            NPU_cur_freq=$(cat /sys/kernel/debug/clk/scmi_clk_npu/clk_rate)
        fi
        # 比较设置的 NPU 频率和当前 NPU 频率
        print_and_compare_result $NPU_freq $NPU_cur_freq

        echo "DDR: Setting frequency"
        if [ -e /sys/class/devfreq/dmc/governor ];then
            # 如果存在该文件，将 DDR 频率调节策略设置为 userspace
            echo userspace > /sys/class/devfreq/dmc/governor
            # 设置 DDR 频率
            echo $DDR_freq > /sys/class/devfreq/dmc/userspace/set_freq
            # 等待设置生效
            sleep 0.5
            # 获取当前 DDR 频率
            DDR_cur_freq=$(cat /sys/class/devfreq/dmc/cur_freq)
        else
            # 从内核调试信息中获取 DDR 频率
            DDR_cur_freq=$(cat /sys/kernel/debug/clk/clk_summary | grep scmi_clk_ddr | awk '{split($0,a," "); print a[5]}')
        fi
        # 比较设置的 DDR 频率和当前 DDR 频率
        print_and_compare_result $DDR_freq $DDR_cur_freq
        ;;

    *)
        # 如果设置策略不在上述范围内，输出设置策略未实现的信息
        echo "Setting strategy not implemented"
        exit 1
        ;;
esac

# 将频率设置和查询结果写入 freq_set_status 文件
echo "Frequency settings:" > ./freq_set_status
echo "CPU: "$CPU_freq >> ./freq_set_status
echo "NPU: "$NPU_freq >> ./freq_set_status
echo "DDR: "$DDR_freq >> ./freq_set_status
echo "Current frequencies:" >> ./freq_set_status
echo "CPU: "$CPU_cur_freq >> ./freq_set_status
echo "NPU: "$NPU_cur_freq >> ./freq_set_status
echo "DDR: "$DDR_cur_freq >> ./freq_set_status
if [ $freq_set_status == 0 ];then
    # 如果频率设置状态为 0，表示设置成功，写入相应信息
    echo "All settings successful" >> ./freq_set_status
    echo -e "\033[32mSUCCESS: All components set to maximum frequency\033[0m"
else
    # 否则表示设置失败，写入相应信息
    echo "Some settings failed" >> ./freq_set_status
    echo -e "\033[31mWARNING: Not all components were set to maximum frequency\033[0m"
fi    
