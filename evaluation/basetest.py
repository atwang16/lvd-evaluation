import subprocess
import sys
import os.path
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
IMG_DB_PATH = os.path.join(ROOT_PATH, "datasets")
BASE_TEST_BUILD = os.path.join(ROOT_PATH, "build_basetest", "basetest")
PARAMETERS_FILE_PATH = os.path.join(ROOT_PATH, "evaluation", "parameters.txt")
DESCRIPTOR_PARAMETERS_DIR = os.path.join(ROOT_PATH, "descriptors")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
distance = "L2"

def is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext in IMG_EXTENSIONS

def generate_results(sf):
    img_num = 2
    image_sf_path = os.path.join(img_database_path, sf)
    results_sf_path = os.path.join(res_database_path, sf)
    expand_image_sf = os.listdir(image_sf_path)

    images = []
    for file in expand_image_sf:
        if is_image(file):
            images.append(file)
    sorted(images)

    img1_path = os.path.join(image_sf_path, images[0])
    img1_base_name, _ = os.path.splitext(images[0])
    desc1_path = os.path.join(results_sf_path, img1_base_name + "_ds.csv")
    kp1_path = os.path.join(results_sf_path, img1_base_name + "_kp.csv")

    while img_num <= len(images):
        current_img = images[img_num - 1]
        img_base_name, _ = os.path.splitext(current_img)
        current_ds = img_base_name + "_ds.csv"
        current_kp = img_base_name + "_kp.csv"
        current_hom = "H1to" + str(img_num) + "p"

        img2_path = os.path.join(image_sf_path, current_img)
        desc2_path = os.path.join(results_sf_path, current_ds)
        kp2_path = os.path.join(results_sf_path, current_kp)
        hom_path = os.path.join(image_sf_path, current_hom)

        args = [BASE_TEST_BUILD, PARAMETERS_FILE_PATH, desc_name, img1_path, desc1_path, kp1_path, img2_path, desc2_path,
                kp2_path, hom_path, distance, results_sf_path + os.sep]

        print(desc_name + ": " + database + " - " + sf + ": " + images[0] + " to " + current_img)
        subprocess.run(args)
        img_num += 1

####### MAIN #######

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python basetest.py desc_name database_name")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]
    img_database_path = os.path.join(IMG_DB_PATH, database)
    res_database_path = os.path.join(RESULTS_PATH, desc_name, database)

    with open(os.path.join(DESCRIPTOR_PARAMETERS_DIR, desc_name + "_parameters.txt")) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if(value[-1] == "\n"):
                value = value[:-1]
            if var == "distance":
                distance = value
            line = f.readline()

    subfolders = os.listdir(res_database_path)

    for sf in subfolders:
        if os.path.isdir(os.path.join(res_database_path, sf)):
            generate_results(sf)