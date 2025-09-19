from preprocessor import ImagePreprocessor
import cv2
import numpy as np
import time


def default_process(img, identity):
    """
    默认处理流程：非均匀校正 -> 盲元检测与补偿 -> 拉伸 -> 16位数据
    """
    if identity == "mwir_fix":
        nuc_path = "E:/145corr/nuc/biaoding/tanceqi/0902_lwir_corr_14bit_gaintiaozheng/nuc/lwir/linear/corr_0.56ms_multi_calibrate_10_20_30/multi_point_calib.npz"
        bp_path = "E:/145corr/nuc/biaoding/tanceqi/0902_lwir_corr_14bit_gaintiaozheng/bp/lwir/0.56ms/blind_pixels.npz"
        img_path = "E:/145corr/nuc/waichang/0830/mwir_building_dandian/0830_11ma_mwir_building_raw.png"
    elif identity == "lwir_fix":
        nuc_path = "E:/145corr/nuc/biaoding/tanceqi/0902_lwir_corr_14bit_gaintiaozheng/nuc/lwir/linear/corr_0.56ms_multi_calibrate_10_20_30/multi_point_calib.npz"
        bp_path = "E:/145corr/nuc/biaoding/tanceqi/0902_lwir_corr_14bit_gaintiaozheng/bp/lwir/0.56ms/blind_pixels.npz"
        img_path = "E:/145corr/nuc/waichang/0830/lwir_0830_raw/raw_0t_0.56ms_20240825_195244_0000.png"

    nuc_para = np.load(nuc_path)
    a_map, b_map, global_a, global_b = nuc_para["a_map"], nuc_para["b_map"], nuc_para["ga"], nuc_para["gb"]
    bp_para = np.load(bp_path)["blind"].astype(bool)

    frame = cv2.imread(img_path, -1)

    max_val = 4095 if identity == "mwir_fix" else 16383

    # 创建 ImagePreprocessor 实例
    preprocessor = ImagePreprocessor()

    t1 = time.time()
    process_res = frame
    if identity == "swir_fix":
        process_res = preprocessor.apply_dw_nuc(
            frame, 0, 0,  0,  0
        )

    elif identity in ["lwir_fix", "mwir_fix"]:
        process_res = preprocessor.apply_linear_calibration(
            frame,
             a_map,
             b_map,
             global_a,
             global_b,
             max_val,
        )

    process_res = preprocessor.compensate_with_filter(process_res, bp_para)

    process_res = preprocessor.stretch_u16(process_res, max_val, downsample=2)

    process_res = preprocessor.apply_sharping(process_res)

    t2 = time.time()
    print(f"default_process 处理耗时: {t2 - t1:.4f} 秒")

    return process_res


if __name__ == "__main__":
    default_process(0, "lwir_fix")