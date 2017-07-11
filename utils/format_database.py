import subprocess
import sys
import os.path
import os

root_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
datasets_folder = os.path.join(root_folder, "datasets")
sep = "_"
img_extensions = [".jpg", ".png", ".ppm", ".pgm"]
preprocessing = False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python imageretrieval.py path_to_database")
        sys.exit(1)

    db_path = sys.argv[1]
    db_name = db_path.split(os.sep)[-1]
    db_prefix = db_name[0:min(3, len(db_name))]
    if os.sep not in db_path:
        db_path = os.path.join(datasets_folder, db_name)

    print("The database will be renamed with the following file conventions:")

    subdirs = os.listdir(db_path)
    sdir_num = 1
    subdirs_old = []
    subdirs_new = []
    images_old = []
    images_new = []
    for sdir in subdirs:
        if os.path.isdir(os.path.join(db_path, sdir)):
            if preprocessing:
                sdir_rename = sdir[4:] # preprocessing step, if necessary; modify this line
                os.rename(os.path.join(db_path, sdir), os.path.join(db_path, sdir_new))
                sdir = sdir_rename

            sdir_num_fmt = "{0:03d}".format(sdir_num)
            subdirs_old.append(os.path.join(db_path, sdir))
            subdirs_new.append(os.path.join(db_path, db_prefix + sep + sdir_num_fmt + sep + sdir))

            images = sorted(os.listdir(os.path.join(db_path, sdir)))

            images_new.append([])
            images_old.append([])
            img_num = 1
            for img in images:
                img_path = os.path.join(db_path, sdir, img)
                img_ext = os.path.splitext(img_path)[1]
                if img_ext in img_extensions:
                    if preprocessing:
                        img_rename = img[4:] # preprocessing step, if necessary; modify this line
                        os.rename(os.path.join(db_path, sdir, img), os.path.join(db_path, sdir, img_rename))
                        img = img_rename

                    img_num_fmt = "{0:03d}".format(img_num)
                    images_old[-1].append(os.path.join(subdirs_new[-1], img))
                    images_new[-1].append(os.path.join(subdirs_new[-1], db_prefix + sep + sdir_num_fmt + sep + img_num_fmt + img_ext))
                    img_num += 1
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



