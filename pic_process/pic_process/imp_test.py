import rgb_dis2

if __name__ == "__main__":
    # 用户可以在此处设置参数
    input_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\word_200"  # 输入文件夹路径
    output_folder = r"F:\FPGA\unilumin_work\pic\1230\2"  # 输出文件夹路径
    scale_factor = 20                  # 缩放比例因子，用户可自行调整

    rgb_dis2.batch_process(input_folder, output_folder, scale_factor)