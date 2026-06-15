import argparse
import os
import shutil


def build_loso_split(data_folder, output_folder, num_subjects=26, num_classes=5):
    os.makedirs(output_folder, exist_ok=True)

    for sub_num in range(1, num_subjects + 1):
        sub_prefix = f"sub{sub_num:02d}"
        sub_folder = os.path.join(output_folder, sub_prefix)
        os.makedirs(sub_folder, exist_ok=True)

        for class_id in range(num_classes):
            class_path = os.path.join(data_folder, str(class_id))
            if not os.path.exists(class_path):
                continue

            files = [f for f in os.listdir(class_path) if f.startswith(sub_prefix)]
            not_files = [f for f in os.listdir(class_path) if not f.startswith(sub_prefix)]

            if files:
                test_folder = os.path.join(sub_folder, "test", str(class_id))
                os.makedirs(test_folder, exist_ok=True)
                for fname in files:
                    shutil.copy(os.path.join(class_path, fname), os.path.join(test_folder, fname))

            train_folder = os.path.join(sub_folder, "train", str(class_id))
            os.makedirs(train_folder, exist_ok=True)
            for fname in not_files:
                shutil.copy(os.path.join(class_path, fname), os.path.join(train_folder, fname))

    print("LOSO split saved to:", output_folder)


def parse_args():
    parser = argparse.ArgumentParser(description="Build CASME II leave-one-subject-out directory splits.")
    parser.add_argument("--data-folder", required=True, help="merged class-wise data directory")
    parser.add_argument("--output-folder", required=True, help="output LOSO directory")
    parser.add_argument("--num-subjects", type=int, default=26)
    parser.add_argument("--num-classes", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_loso_split(args.data_folder, args.output_folder, args.num_subjects, args.num_classes)
