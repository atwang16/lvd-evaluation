import subprocess
import sys
import os.path
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
datasets_folder = None
img_extensions = [".jpg", ".png", ".ppm", ".pgm"]
preprocessing = True 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python imageretrieval.py database")
        sys.exit(1)

    with open(os.path.join(ROOT_PATH, "project_structure.txt")) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            dir = line_split[-1]
            if(dir[-1] == "\n"):
                dir = dir[:-1]
            if var == "DATASETS_FOLDER":
                datasets_folder = os.path.join(ROOT_PATH, dir)
            line = f.readline()

    if datasets_folder is None:
        print("Error: could not find all of the necessary directory paths from the following file:")
        print(" ", os.path.join(ROOT_PATH, "project_structure.txt"))
        sys.exit(1)

    db_path = sys.argv[1]
    db_name = db_path.split(os.sep)[-1]
    db_prefix = db_name[0:min(3, len(db_name))]
    if os.sep not in db_path:
        db_path = os.path.join(datasets_folder, db_name)

    print("The database will be renamed with the following file conventions:")

    subdirs = sorted(os.listdir(db_path))
    sdir_num = 1
    subdirs_old = []
    subdirs_new = []
    images_old = []
    images_new = []

    for sdir in subdirs:
        if os.path.isdir(os.path.join(db_path, sdir)):
            if preprocessing:
                sdir_rename = sdir # preprocessing step, if necessary; modify this line
                os.rename(os.path.join(db_path, sdir), os.path.join(db_path, sdir_rename))
                sdir = sdir_rename

            sdir_num_fmt = "{0:03d}".format(sdir_num)
            subdirs_old.append(os.path.join(db_path, sdir))
            subdirs_new.append(os.path.join(db_path, db_prefix + "_" + sdir_num_fmt + "_" + sdir))

            images = sorted(os.listdir(os.path.join(db_path, sdir)))

            images_new.append([])
            images_old.append([])
            img_num = 1
            for img in images:
                img_path = os.path.join(db_path, sdir, img)
                img_ext = os.path.splitext(img_path)[1]
                if img_ext in img_extensions:
                    if preprocessing:
                        img_rename = img # preprocessing step, if necessary; modify this line
                        os.rename(os.path.join(db_path, sdir, img), os.path.join(db_path, sdir, img_rename))
                        img = img_rename

                    img_num_fmt = "{0:03d}".format(img_num)
                    images_old[-1].append(os.path.join(subdirs_new[-1], img))
                    images_new[-1].append(os.path.join(subdirs_new[-1], db_prefix + "_" + sdir_num_fmt + "_" + img_num_fmt + img_ext))
                    img_num += 1
                elif img[0] == "H": # homography
                    if preprocessing:
                        hom = img
                        hom_rename = "H" + hom[2] + "to" + hom[4] + "p"
                        os.rename(os.path.join(db_path, sdir, hom), os.path.join(db_path, sdir, hom_rename))
                        hom = hom_rename
            sdir_num += 1

    for i in range(len(subdirs_new)):
        print(" - ", subdirs_new[i])
        for img in images_new[i]:
            print("   - ", img)

    yn = input("Are you sure you would like to proceed? This operation cannot be undone. [y/n]: ")
    if yn == "y":
        for i in range(len(subdirs_old)):
            os.rename(subdirs_old[i], subdirs_new[i])
            for j in range(len(images_old[i])):
                os.rename(images_old[i][j], images_new[i][j])



