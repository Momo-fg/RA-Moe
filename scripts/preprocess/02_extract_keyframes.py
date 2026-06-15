import os
import shutil
import pandas as pd
import argparse

def read_annotations(annotation_file):
    df = pd.read_excel(annotation_file)
    return df

def process_data(source_dir, target_dir, annotations):
    for _, row in annotations.iterrows():
        subject = f"sub{row['Subject']:02d}"  # 参与者编号
        filename = row["Filename"]           # 视频片段名称
        emotion = row["Estimated Emotion"]   # 情绪类别
        
        # 构建原始路径
        source_video_folder = os.path.join(source_dir, subject, filename)
        
        # 构建目标路径
        target_category_folder = os.path.join(target_dir, emotion)
        os.makedirs(target_category_folder, exist_ok=True)
        
        # 获取关键帧编号
        onset_frame_num = row['OnsetFrame']
        apex_frame_num = row['ApexFrame']
        offset_frame_num = row['OffsetFrame']
        
        # 处理关键帧
        key_frames = {
            "onset": (onset_frame_num, f"{subject}_{filename}_onset.jpg"),
            "apex": (apex_frame_num, f"{subject}_{filename}_apex.jpg"),
            "offset": (offset_frame_num, f"{subject}_{filename}_offset.jpg")
        }
        
        # 如果 Apex 帧与 Onset 或 Offset 帧重合，需要特殊处理
        if apex_frame_num == onset_frame_num:
            print(f"Warning: Apex frame ({apex_frame_num}) overlaps with Onset frame for {subject}/{filename}.")
            key_frames["onset"] = (onset_frame_num, f"{subject}_{filename}_onset.jpg")  # 保存为 onset.jpg
            key_frames["apex"] = (onset_frame_num, f"{subject}_{filename}_apex.jpg")    # 同时保存为 apex.jpg
        if apex_frame_num == offset_frame_num:
            print(f"Warning: Apex frame ({apex_frame_num}) overlaps with Offset frame for {subject}/{filename}.")
            key_frames["offset"] = (offset_frame_num, f"{subject}_{filename}_offset.jpg")  # 保存为 offset.jpg
            key_frames["apex"] = (offset_frame_num, f"{subject}_{filename}_apex.jpg")      # 同时保存为 apex.jpg
        
        # 移动并重命名关键帧
        for frame_num, dst_name in key_frames.values():
            src_frame = f"img{frame_num}.jpg"
            src_path = os.path.join(source_video_folder, src_frame)
            dst_path = os.path.join(target_category_folder, dst_name)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: File {src_path} not found.")

def main(annotation_file, source_dir, target_dir):
    annotations = read_annotations(annotation_file)
    process_data(source_dir, target_dir, annotations)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CASME II onset/apex/offset frames.")
    parser.add_argument("--annotation-file", required=True, help="CASME II annotation xlsx")
    parser.add_argument("--source-dir", required=True, help="cropped CASME II sequence root")
    parser.add_argument("--target-dir", required=True, help="output keyframe directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.annotation_file, args.source_dir, args.target_dir)
