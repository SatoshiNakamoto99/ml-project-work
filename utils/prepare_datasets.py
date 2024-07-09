#TODO: INSERIRE FUNZIONI PER SCARICARE E CREARE I DATASET
import os
import gdown
import random
import shutil
import glob
import tqdm


def download_google_file(shader_url, output_name):
  id_url = "https://drive.google.com/uc?id=" + shader_url.split("/")[5]
  gdown.download(id_url, output_name)

def create_folder_structure(parent):
    # Crea la cartella "parent" se non esiste
    if not os.path.exists(parent):
        os.makedirs(parent)

    # Crea la cartella "TRAINING_SET" dentro la cartella "GT" se non esiste
    if not os.path.exists(os.path.join(parent, "TRAINING_SET")):
        os.makedirs(os.path.join(parent, "TRAINING_SET"))

    # Crea la cartella "0" e "1" dentro la cartella "TRAINING_SET" in "GT" se non esistono
    for folder in ["0", "1"]:
        folder_path = os.path.join(parent, "TRAINING_SET", folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def random_file(source_folder,destination_folder, num, flag = False):
  mp4_files = [f for f in os.listdir(source_folder) if f.endswith(".mp4")]
  selected_files = random.sample(mp4_files, num)

  for file_name in selected_files:
     file_path = os.path.join(source_folder, file_name)
     if(flag == False):
      shutil.copy(file_path, destination_folder)
     else:
      shutil.move(file_path, destination_folder)
  return selected_files

def create_test_structure(parent):
      # Crea la cartella "TEST_SET" dentro la cartella "parent" se non esiste
    if not os.path.exists(os.path.join(parent, "TEST_SET")):
      os.makedirs(os.path.join(parent, "TEST_SET"))
    else:
      # eliminala e ricreala
      shutil.rmtree(os.path.join(parent, "TEST_SET"))
      os.makedirs(os.path.join(parent, "TEST_SET"))


    # Crea la cartella "0" e "1" dentro la cartella "TEST_SET" in "parent" se non esistono
    for folder in ["0", "1"]:
        folder_path = os.path.join(parent, "TEST_SET", folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def calculate_0_prob(folder):
  folder = folder+"/VIDEOS/TRAINING_SET"
  num_fire = 0
  for filename in os.listdir(folder+"/1"):
        if filename.endswith(".mp4"):
            num_fire += 1

    # Conta il numero di .mp4 nell  "0" folder
  num_no_fire = 0
  for filename in os.listdir(folder+"/0"):
        if filename.endswith(".mp4"):
            num_no_fire += 1
  return num_no_fire/(num_fire+num_no_fire), num_no_fire

def create_test_set(folder, dim, prob_a_priori = False):
  if(dim>1):
    print("dim dev'essere minore di 1. Esempio (20% --> 0.2)")
  else:
    prob_no_fire, num_no_fire = calculate_0_prob(folder)
    num_tot_test = (num_no_fire/prob_no_fire )*dim
    if (prob_a_priori == False):
      num_sample_test_0 =  round(num_tot_test * prob_no_fire)
      num_sample_test_1 = num_sample_test_0
    else:
      num_sample_test_0 = round(num_tot_test * prob_no_fire)
      num_sample_test_1 = round(num_tot_test * (1-prob_no_fire))
    initial_path = os.getcwd()
    os.chdir(folder)
    create_test_structure("GT")
    create_test_structure("VIDEOS")
    os.chdir(initial_path)
    video_training = "/VIDEOS/TRAINING_SET"
    video_test = "/VIDEOS/TEST_SET"
    rtf_training = "/GT/TRAINING_SET"
    rtf_test = "/GT/TEST_SET"

    folder_path_0_mp4 = folder+video_training+"/0"
    folder_path_1_mp4 = folder+video_training+"/1"
    folder_path_0_rtf = folder+rtf_training+"/0"
    folder_path_1_rtf = folder+rtf_training+"/1"
    test_path_0_mp4 = folder+video_test+"/0"
    test_path_1_mp4 = folder+video_test+"/1"
    test_path_0_rtf = folder+rtf_test+"/0"
    test_path_1_rtf = folder+rtf_test+"/1"

    selected_file = random_file(folder_path_0_mp4,test_path_0_mp4, num_sample_test_0, True)
    for file_name in selected_file:
      file_path = os.path.join(folder_path_0_rtf, file_name[:-4]+".rtf")
      shutil.move(file_path, test_path_0_rtf)

    selected_file = random_file(folder_path_1_mp4,test_path_1_mp4, num_sample_test_1, True)
    for file_name in selected_file:
      file_path = os.path.join(folder_path_1_rtf, file_name[:-4]+".rtf")
      shutil.move(file_path, test_path_1_rtf)

def copy_folder(src_folder, dest_folder):
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    shutil.copytree(src_folder, dest_folder)

def extract_frames(selected_dataset, videos_path, frames_path):
    current_directory = os.getcwd()
    os.chdir("datasets/"+selected_dataset+"/VIDEOS")
    file_list = [path for path in glob.glob(os.path.join(videos_path,"**"), recursive=True)
                if os.path.isfile(path)]
    print(videos_path)
    print(file_list)
    for video in tqdm.tqdm(file_list):
        if os.path.isdir(os.path.join(frames_path, video)):
            continue

        os.makedirs(os.path.join(frames_path, video))
        #extract_frames(video)
        os.system("ffmpeg -i {} -r 1/1 {}/{}/$Frame{}.png".format(video, frames_path, video, "%05d")) #estrae un frame ogni secondo
        
    os.chdir(current_directory)
    # specifica il percorso della cartella da spostare
    source = "datasets/"+selected_dataset+"/VIDEOS/FRAMES"
    # specifica il percorso di destinazione della cartella
    destination = "datasets/"+selected_dataset
    # sposta la cartella dal percorso di origine al percorso di destinazione
    shutil.move(source, destination)

def copy_content(source_dir, destination_dir):
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        destination_item = os.path.join(destination_dir, item)
        shutil.copy(source_item, destination_item)