# -*- coding: utf-8 -*-


import os
import pydicom
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from scipy.ndimage import zoom
from skimage.transform import resize
from tqdm import tqdm


preprocess_BASEPATH = "/home/ubuntu/workspace"
all_data_folder = os.path.join(preprocess_BASEPATH, "dataset", "017.치매진단뇌영상", "01.데이터", "1.DICOM", "training", "원시데이터")
h5_after_path = os.path.join(preprocess_BASEPATH, "h5_after")
csv_path = r"/home/ubuntu/workspace/dataset/017.치매진단뇌영상/01.데이터/1.DICOM/meta/MRI_dataset_final.csv"


# csv load
try:
    my_csv = pd.read_csv(csv_path, encoding="euc-kr", header=0)
    # print(my_csv.head())
    scan_id = "subject_001"
    # print(my_csv.columns.tolist())
    diag = my_csv.loc[my_csv["MRI convert 익명코드"] == scan_id, "3T_DATA진단명 "]
    # print(diag)
    

except Exception as e:
    print(e)
    my_csv = { # 가짜 데이터
        "diagnosis": {
            "I123123": "MCI",
            "I75356": "AD",
            "D12345": "CN",
        }
    }
    print("Failed to read CSV.")


def get_parent_folders(path):
    parent_folders = set()
    for root, dirs, filenames in tqdm(os.walk(path)):
        # print(">>>>", root)
        for file_name in filenames:
            if file_name.endswith(".dcm"):
                parent_folders.add(root)
    return list(parent_folders)


def load_dicom_volume_parallel(folder):
    dicom_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".dcm")
    ]

    with ThreadPoolExecutor(max_workers=6) as executor:
        dicom_data_list = list(executor.map(pydicom.dcmread, dicom_files))

    dicom_data_list.sort(key=lambda d: int(d.InstanceNumber))

    slices = [d.pixel_array for d in dicom_data_list]
    spacing = [dicom_data_list[0].SliceThickness] + list(dicom_data_list[0].PixelSpacing)

    return np.stack(slices, axis=0), spacing

    # # InstanceNumber를 기준으로 정렬
    # dicom_files_sorted = []
    # try:
    #     for idx, path in enumerate(dicom_files):
    #         dicom_data = pydicom.dcmread(path)
    #         dicom_files_sorted.append((idx, int(dicom_data.InstanceNumber)))
    #     dicom_files_sorted.sort(key=lambda x: x[1])
    # except:
    #     raise Exception("InstanceNumber Does Not Exist.")

    # # DICOM 파일에서 픽셀 데이터 읽기
    # slices = []
    # spacing = None
    # for idx, _ in dicom_files_sorted:
    #     dicom_data = pydicom.dcmread(dicom_files[idx])
    #     slices.append(dicom_data.pixel_array)
    #     z_spacing = (
    #         dicom_data.SpacingBetweenSlices
    #         if hasattr(dicom_data, "SpacingBetweenSlices")
    #         else dicom_data.SliceThickness
    #     )
    #     x_spacing, y_spacing = dicom_data.PixelSpacing
    #     spacing = [z_spacing, x_spacing, y_spacing]

    # # 3D 스캔 배열 생성
    # volume = np.stack(slices, axis=0)

    # return volume, spacing


def resample_volume(
    volume, original_spacing, target_shape=(800, 800, 800), final_shape=(224, 224, 224)
):
    """
    볼륨 데이터를 original_spacing을 고려하여 재샘플링 후 zero padding, 정규화 및 최종 리사이징 수행.

    Args:
        volume (numpy array): 원본 3D MRI 데이터.
        original_spacing (numpy array): 원본 볼륨의 해상도 정보.
        target_shape (tuple): 중간 목표 크기 (기본값: (300, 300, 300)).
        final_shape (tuple): 최종 출력 크기 (기본값: (224, 224, 224)).

    Returns:
        final_volume (numpy array): (224, 224, 224) 크기로 변환된 볼륨 데이터.
    """

    if volume[0].shape[0] > 600:
        print("more than 600")
    # MRI 데이터의 해상도를 고려한 보간 (Resampling)
    zoom_factors = [1 / original_spacing[i] for i in range(3)]
    resampled_volume = zoom(volume, zoom=zoom_factors, order=1)  # 선형 보간 사용

    (z_dim, x_dim, y_dim) = resampled_volume.shape
    (Z, X, Y) = target_shape  # (300, 300, 300) 가정

    # 초과한다면 중앙 부분만 크롭
    z_start = (z_dim - Z) // 2 if z_dim > Z else 0
    z_end = z_start + Z if z_dim > Z else z_dim
    x_start = (x_dim - X) // 2 if x_dim > X else 0
    x_end = x_start + X if x_dim > X else x_dim
    y_start = (y_dim - Y) // 2 if y_dim > Y else 0
    y_end = y_start + Y if y_dim > Y else y_dim

    resampled_volume = resampled_volume[z_start:z_end, x_start:x_end, y_start:y_end]

    epsilon = 1e-8  # 분모가 0이 되는 것을 방지
    min_val = np.min(resampled_volume)
    max_val = np.max(resampled_volume)

    if max_val - min_val > epsilon:  # 0으로 나누는 것 방지
        resampled_volume = (
            (resampled_volume - min_val) / (max_val - min_val + epsilon) * 255
        )

    # uint8 변환 (이미지 데이터 처리에 적합)
    resampled_volume = resampled_volume.astype(np.uint8)

    # Zero Padding 적용 (중앙 정렬)
    padded_volume = np.zeros(target_shape, dtype=np.float32)

    # 기존 데이터 삽입 (중앙 정렬)
    z_start = (target_shape[0] - resampled_volume.shape[0]) // 2
    x_start = (target_shape[1] - resampled_volume.shape[1]) // 2
    y_start = (target_shape[2] - resampled_volume.shape[2]) // 2

    padded_volume[
        z_start : z_start + resampled_volume.shape[0],
        x_start : x_start + resampled_volume.shape[1],
        y_start : y_start + resampled_volume.shape[2],
    ] = resampled_volume

    # 최종 (224, 224, 224)로 리사이징
    final_volume = resize(
        padded_volume, final_shape, order=1, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)

    # visualize_slices(volume, final_volume)

    return final_volume



def visualize_slices(volume, final_volume):
    """
    원본 볼륨과 최종 변환된 볼륨의 중앙 슬라이스 6개를 시각화 (2x3 subplot).

    Args:
        volume (numpy array): 원본 (300, 300, 300) 볼륨 데이터.
        final_volume (numpy array): 최종 변환된 (224, 224, 224) 볼륨 데이터.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    if volume is not None:
        # 원본 볼륨 중앙 슬라이스 (z, x, y 축 기준)
        z_mid, x_mid, y_mid = (
            volume.shape[0] // 2,
            volume.shape[1] // 2,
            volume.shape[2] // 2,
        )
        slices_original = [
            volume[z_mid, :, :],
            volume[:, x_mid, :],
            volume[:, :, y_mid],
        ]
        titles_original = ["Original Z-Slice", "Original X-Slice", "Original Y-Slice"]

        # 원본 MRI 슬라이스 표시
        for i in range(3):
            im = axes[0, i].imshow(slices_original[i], cmap="gray")
            axes[0, i].set_title(titles_original[i])
            fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)  # Colorbar 추가
            print(slices_original[i].shape, end=" ")

    print()

    # 최종 볼륨 중앙 슬라이스 (z, x, y 축 기준)
    z_mid_f, x_mid_f, y_mid_f = (
        final_volume.shape[0] // 2,
        final_volume.shape[1] // 2,
        final_volume.shape[2] // 2,
    )
    slices_final = [
        final_volume[z_mid_f, :, :],
        final_volume[:, x_mid_f, :],
        final_volume[:, :, y_mid_f],
    ]
    titles_final = ["Final Z-Slice", "Final X-Slice", "Final Y-Slice"]

    # 변환된 MRI 슬라이스 표시
    for i in range(3):
        im = axes[1, i].imshow(slices_final[i], cmap="gray")
        axes[1, i].set_title(titles_final[i])
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        print(slices_final[i].shape, end=" ")


    plt.tight_layout()
    plt.show()

def save_h5_file(h5_filename, data):
    os.makedirs(os.path.dirname(h5_filename), exist_ok=True)
    with h5py.File(h5_filename, "w") as hf:
        hf.create_dataset("volume", data=data, compression="gzip")

def get_MRI_class_from_scanId(csv, scan_id):
    try:
        scan_class = csv.loc[csv["MRI convert 익명코드"] == scan_id, "3T_DATA진단명 "].values[0]
        if scan_class in ["NC"]:
            return "CN"
        elif scan_class in ["AMCI", "aMCI"]:
            return "MCI"
        elif scan_class in ["AD", "ADD"]:
            return "AD"
        else:
            raise Exception("No class found.")
    except:
        raise Exception("Code error in 'get_MRI_class_from_scanId'")


def process_folder(folder):
    scan_id = os.path.basename(folder)

    try:
        dcm_class = get_MRI_class_from_scanId(my_csv, scan_id[:-3])
        volume, spacing = load_dicom_volume_parallel(folder)
        after_volume = resample_volume(volume, spacing)

        h5_filename = os.path.join(h5_after_path, dcm_class, f"{scan_id}.h5")
        save_h5_file(h5_filename, after_volume)

        return (scan_id, "Success")
    except Exception as e:
        return (scan_id, f"Failed: {str(e)}")
    # try:
    #     h5_filename = os.path.join(h5_after_path, dcm_class, f"{scan_id}.h5")
    #     with h5py.File(h5_filename, "w") as hf:
    #         hf.create_dataset("volume", data=after_volume)
    # except Exception as e:
    #     process_failed_dirs.append([scan_id, e])
    #     print("Failed to process in step 2.", e)
    #     continue

parent_folders = get_parent_folders(all_data_folder)
print("부모폴더 개수:", len(parent_folders))

with Pool(processes=6) as pool:
    results = list(tqdm(pool.imap(process_folder, parent_folders), total=len(parent_folders)))

print("Preprocess Done.")
print("Failed:", [res for res in results if "Failed" in res])

# for folder in tqdm(parent_folders):
#     scan_id = os.path.basename(folder)

#     try:
#         dcm_class = get_MRI_class_from_scanId(my_csv, scan_id[:-3])
#         volume, spacing = load_dicom_volume(folder)
#         after_volume = resample_volume(volume, spacing)
#     except Exception as e:
#         process_failed_dirs.append([scan_id, e])
#         print("Failed to process in step 1.")
#         continue

#     try:
#         h5_filename = os.path.join(h5_after_path, dcm_class, f"{scan_id}.h5")
#         with h5py.File(h5_filename, "w") as hf:
#             hf.create_dataset("volume", data=after_volume)
#     except Exception as e:
#         process_failed_dirs.append([scan_id, e])
#         print("Failed to process in step 2.", e)
#         continue

#     # visualize_slices(volume, after_volume)
#     # print(f"{h5_filename} complete.")

# print("Preprocess Done.")
# print("Failed:", process_failed_dirs)


# After Only Visualization
with h5py.File(os.path.join(h5_after_path, "AD", "subject_002_t1.h5"), 'r') as hf:
    after_volume = hf["volume"]
    visualize_slices(None, after_volume)

