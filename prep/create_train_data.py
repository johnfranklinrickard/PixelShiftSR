import utility.util as util
import prep.image_snippet as snippet
import prep.downscale as ds
import os
import datetime
import numpy


def div2k_name(number):
    return os.path.join(div2k_dir, f"{number:04d}.png")


def extract_origin_img(image, padded):
    origin_image, x, y = snippet.extract_random_snippet(image)
    util.append_to_file(snipped_coords_txt, f"{x} {y}\n")  # save coordinates in file
    origin_name = os.path.join(origin_dir, "original-" + padded)
    util.save_array_image(origin_image, origin_name)
    return origin_image


def create_control_img(image, padded_name):
    control_img = ds.batch_shift_downscale(image, snippet.no_shift())
    control_name = os.path.join(control_dir, "control-" + padded_name)
    util.save_array_image(control_img, control_name)


def create_half_img(image, padded_name):
    half_img = ds.batch_shift_downscale(image, snippet.half_shift())
    half_name = os.path.join(half_dir, "half-" + padded_name)
    util.save_array_image(half_img, half_name)


def create_quarter_img(image, padded_name):
    quarter_img = ds.batch_shift_downscale(image, snippet.quarter_shift())
    quarter_name = os.path.join(quarter_dir, "quarter-" + padded_name)
    util.save_array_image(quarter_img, quarter_name)


def create_random0_img(image, padded_name):
    random0_img = ds.batch_shift_downscale(image, snippet.random0_shifts())
    random0_name = os.path.join(random0_dir, "random0-" + padded_name)
    util.save_array_image(random0_img, random0_name)


def create_dataset(start, stop, samples_per_img):
    print(f"{datetime.datetime.now().time()}: Start")

    util.create_file(snipped_coords_txt, f"start={start}\nstop={stop}\nsamples={samples_per_img}\n")

    image_number = 0
    for i in range(start, stop):
        img = util.load_img_ndarray(div2k_name(i))

        for j in range(samples_per_img):
            img_name = f"{image_number:05d}.png"
            image_number += 1

            origin_img = extract_origin_img(img, img_name)

            create_control_img(origin_img, img_name)
            create_half_img(origin_img, img_name)
            create_quarter_img(origin_img, img_name)
            create_random0_img(origin_img, img_name)

        print(f"{datetime.datetime.now().time()}: {i} done")
    print(f"{datetime.datetime.now().time()}: Finished")


def prepare_directory(folder_name, shifts, step=1.0):
    util.create_dir(folder_name)
    util.create_file(f"{folder_name}/{x_coords_txt}", x_coords(step))
    util.create_file(f"{folder_name}/{y_coords_txt}", y_coords(shifts, step))


def x_coords(step):
    text = ""
    pixel_count = low_res_width / step
    for x in numpy.arange(0.0, low_res_width, step):
        text += " ".join([str(x)] * int(pixel_count))
        text += "\n"
    return text


def y_coords(shifts, step):
    text = ""
    shifts_len = len(shifts)
    for x in numpy.arange(0.0, low_res_width, step):
        current_shift = shifts[int(x) % shifts_len]
        text += " ".join(str(i) for i in numpy.arange(current_shift, low_res_width, step))
        text += "\n"
    return text


def prepare():
    prepare_directory(origin_dir, snippet.no_shift(), 0.25)
    prepare_directory(control_dir, snippet.no_shift())
    prepare_directory(half_dir, snippet.half_shift())
    prepare_directory(quarter_dir, snippet.quarter_shift())
    prepare_directory(random0_dir, snippet.random0_shifts())


if __name__ == "__main__":
    x_coords_txt = "x_coords.txt"
    y_coords_txt = "y_coords.txt"
    low_res_width = 64

    # hard coded for train data
    top_folder = "train-data"
    div2k_dir = "../DIV2K Dataset/DIV2K_train_HR"
    origin_dir = f"{top_folder}/original"
    control_dir = f"{top_folder}/control"  # no shift directory
    half_dir = f"{top_folder}/half_shift"
    quarter_dir = f"{top_folder}/quarter_shift"
    random0_dir = f"{top_folder}/random0_shift"
    snipped_coords_txt = f"{origin_dir}/snippet_coords.txt"

    prepare()
    create_dataset(1, 801, 15)  # train_HR

    # hard coded for valid data
    top_folder = "val-data"
    div2k_dir = "../DIV2K Dataset/DIV2K_valid_HR"
    origin_dir = f"{top_folder}/original"
    control_dir = f"{top_folder}/control"  # no shift directory
    half_dir = f"{top_folder}/half_shift"
    quarter_dir = f"{top_folder}/quarter_shift"
    random0_dir = f"{top_folder}/random0_shift"
    snipped_coords_txt = f"{origin_dir}/snippet_coords.txt"

    prepare()
    create_dataset(801, 901, 20)  # valid_HR
