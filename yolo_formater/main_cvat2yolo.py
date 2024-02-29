import os

import shutil
import yaml

from cvat.apps.dataset_manager.util import make_zip_archive

from .split_auto import autosplit
from .split_manual import manualsplit
from .lib_utils_cvat2yolo import (
    create_YOLOv5_folder_tree,
    remove_unwanted_classes,
    transform_cls_labels,
)


def get_datset_classes(names_file, classes_to_keep):
    with open(names_file) as f:
        dataset_names = f.read().splitlines()

    if classes_to_keep == "keep-all":
        return dataset_names
    else:
        print(classes_to_keep)
        classes_to_keep = classes_to_keep.split("|")
        print(classes_to_keep)
        names = [n for n in dataset_names if n in classes_to_keep]
        if len(names) == 0:
            raise ValueError(
                f"--classes arg is not valid, dataset classes: {dataset_names}"
            )
        print(f"KEEPING CLASSES: {names}")
        return names


def form_yaml_file(output_folder, classes):
    number_of_classes = len(classes)
    path = output_folder
    train = os.path.join("Train", "images")
    val = os.path.join("Val", "images")

    with open(f"{output_folder}/{output_folder}.yaml", "w") as stream:
        yaml.dump(
            {
                "path": path,
                "train": train,
                "val": val,
                "nc": number_of_classes,
                "names": classes,
            },
            stream,
            default_flow_style=False,
        )



def convert_to_yolo5(cvat_input_folder, dst_file):
    split = None
    mode = 'manual'
    percentage_empty = 10
    img_format = 'png'
    classes_to_keep = 'keep-all'
    train_folder = 'obj_Train_data'
    val_folder = 'obj_Validation_data'
    test_folder = 'obj_Test_data'
    names_file = "obj.names"
    CVAT_input_folder = cvat_input_folder
    CVAT_work_folder = f"{CVAT_input_folder}_copy"
    names_file_pth = os.path.join(CVAT_work_folder, names_file)
    train_folder = os.path.join(CVAT_work_folder, train_folder)
    val_folder = os.path.join(CVAT_work_folder, val_folder)
    test_folder = os.path.join(CVAT_work_folder, test_folder)
    output_folder = "dataset"
    label_tfrms = None


    shutil.copytree(CVAT_input_folder, CVAT_work_folder)



    assert "." not in img_format, "img_format must be without ."
    assert (
        mode == "autosplit" or mode == "manual"
    ), f"mode must be 'autosplit' or 'manual', {mode} was given"
    if mode == "autosplit":
        assert abs(split) < 1, f"float split (0<split<1) is required, {split} was given"
        assert os.path.exists(
            os.path.join(train_folder)
        ), f"{train_folder} does not exist in {CVAT_work_folder}"
    elif mode == "manual":
        assert (
            os.path.exists(train_folder)
            or os.path.exists(val_folder)
            or os.path.exists(test_folder)
        ), f"At least one of {train_folder}, {val_folder} and {test_folder} must exist"
        if split is not None:
            print("WARNING: skipping split value n manual mode")

    create_YOLOv5_folder_tree(output_folder)

    if label_tfrms is not None:
        transform_cls_labels(CVAT_work_folder, names_file_pth, label_tfrms)

    classes_to_keep = get_datset_classes(names_file_pth, classes_to_keep)
    remove_unwanted_classes(CVAT_work_folder, names_file_pth, classes_to_keep)

    form_yaml_file(output_folder, classes_to_keep)

    if mode == "autosplit":
        autosplit(
            output_folder,
            train_folder,
            val_folder,
            test_folder,
            img_format,
            split,
            percentage_empty,
            lbl_extention="txt",
        )
    elif mode == "manual":
        manualsplit(
            output_folder,
            train_folder,
            val_folder,
            test_folder,
            img_format,
            percentage_empty,
            lbl_extention="txt",
        )


    final_output_path = os.path.join(CVAT_input_folder, output_folder)
    shutil.move(output_folder, final_output_path)
    make_zip_archive(final_output_path, dst_file)



