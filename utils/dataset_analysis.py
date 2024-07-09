import pandas as pd
import json
import os
import shutil
import ffmpeg

def extract_clusters_dataframe(analysis_results_path) -> pd.DataFrame:
    with open(analysis_results_path, "r") as fp:
        raw_data = json.load(fp)
    print(f"Found {len(raw_data)} videos inside clusters")
    df = pd.DataFrame(raw_data)
    df = df[["GroupId", "Path", "Folder"]]
    df["VideoName"] = df["Path"].apply(lambda x: x.split("\\")[-1])
    df["Class"] = df["Folder"].apply(lambda x: x.split("\\")[-1])
    df["Dataset"] = df["Folder"].apply(lambda x: x.split("\\")[-4])
    df = df.sort_values(by=["GroupId", "Class"])
    return df

def show_cluster_dataframe(json_file, verbose = True):
    clusters_data = extract_clusters_dataframe(json_file)
    clusters_data["VideoName"] = clusters_data.groupby(["GroupId","Dataset","Class"])["VideoName"].transform(lambda x: ', '.join(x))
    clusters_data["VideoNumber"] = clusters_data.groupby(["GroupId","Dataset","Class"])["VideoName"].transform(lambda x: len(x))
    clusters_data = clusters_data.drop_duplicates(subset=["GroupId","Dataset","Class"])
    clusters_data = clusters_data.sort_values(["Dataset","Class"]).drop(columns=["Path", "Folder"]).groupby(["GroupId","Dataset","Class"]).sum()
    #clusters_data = clusters_data[["GroupId", "Dataset", "Class", "VideoName"]]
    if verbose == True:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            pd.set_option('display.max_colwidth', None)
            display(clusters_data)
    return clusters_data

def create_dict_from_json(clusters_data):
    datasets_dict = {}
    for dataset in clusters_data.index.get_level_values("Dataset").unique():
        dataset_dict = {}
        dataset_clusters = clusters_data.xs(dataset, level="Dataset")
        for classe in dataset_clusters.index.get_level_values("Class").unique():
            classe_clusters = dataset_clusters.xs(classe, level="Class")
            videos = classe_clusters["VideoName"].str.split(",").explode().str.strip().unique()
            dataset_dict[classe] = list(videos)
        datasets_dict[dataset] = dataset_dict
    return datasets_dict

def create_videos_dict( datasets):
    
    datasets_dict = {}
    for dataset in datasets:
        folder_path = "datasets/analysis/"+dataset+"/VIDEOS/TRAINING_SET"
        dataset_dict = {}
        #dataset_path = os.path.join(folder_path, str(dataset))
        for folder in os.listdir(folder_path):
            if folder in ('0', '1'):
                folder_path_plus = os.path.join(folder_path, folder)
                #print(folder_path)
                class_name = '0' if folder == '0' else '1'
                class_videos = [f for f in os.listdir(folder_path_plus) if os.path.isfile(os.path.join(folder_path_plus, f))]
                dataset_dict[class_name] = class_videos
        datasets_dict[dataset] = dataset_dict
    return datasets_dict

def substract_dict(dict_complete, datasets_dict):
    diff_dict={}

    for dataset, classes_dict in dict_complete.items():
        if dataset in datasets_dict:
            diff_classes_dict = {}
            for classe, videos in classes_dict.items():
                if classe in datasets_dict[dataset]:
                    diff_videos = list(set(videos) - set(datasets_dict[dataset][classe]))
                    if diff_videos:
                        diff_classes_dict[classe] = diff_videos
            if diff_classes_dict:
                diff_dict[dataset] = diff_classes_dict
    return diff_dict

def copy_videos_and_rtf_files(folder_path, false, smoke, fire, file_dict, dest_folder_path):
    for folder_name in os.listdir(folder_path):
        #print(folder_name)
        video_has_fire = fire in folder_name.lower()
        video_has_smoke = smoke in folder_name.lower()
        #print(video_has_fire)
        #print(video_has_smoke)
        if false in folder_name.lower():
            #print(folder_pathname)
            folder_pathname = os.path.join(folder_path, folder_name)
            for video_name in os.listdir(folder_pathname):
                if video_name in file_dict:
                    video_pathname = os.path.join(folder_pathname, video_name)
                    rtf_filename = os.path.splitext(video_name)[0] + ".rtf"
                    rtf_path = os.path.join(dest_folder_path, "GT/TRAINING_SET/0", rtf_filename)
                    if not os.path.exists(rtf_path):
                        # crea il file RTF con il primo campo "0,Fire"
                        with open(rtf_path, "w") as f:
                            f.write("")
                    video_dest_pathname = os.path.join(dest_folder_path+"/VIDEOS/TRAINING_SET/0", video_name)
                    shutil.copy(video_pathname, video_dest_pathname)
        elif video_has_fire:
            folder_pathname = os.path.join(folder_path, folder_name)
            for video_name in os.listdir(folder_pathname):
                if video_name in file_dict:
                    video_pathname = os.path.join(folder_pathname, video_name)
                    rtf_filename = os.path.splitext(video_name)[0] + ".rtf"
                    rtf_path = os.path.join(dest_folder_path, "GT/TRAINING_SET/1", rtf_filename)
                    if not os.path.exists(rtf_path):
                        # crea il file RTF con il primo campo "0,Fire"
                        with open(rtf_path, "w") as f:
                            f.write('0,Fire')
                    video_dest_pathname = os.path.join(dest_folder_path+"/VIDEOS/TRAINING_SET/1", video_name)
                    shutil.copy(video_pathname, video_dest_pathname)
        elif video_has_smoke:
            folder_pathname = os.path.join(folder_path, folder_name)
            for video_name in os.listdir(folder_pathname):
                if video_name in file_dict:
                    video_pathname = os.path.join(folder_pathname, video_name)
                    rtf_filename = os.path.splitext(video_name)[0] + ".rtf"
                    rtf_path = os.path.join(dest_folder_path, "GT/TRAINING_SET/1", rtf_filename)
                    if not os.path.exists(rtf_path):
                        # crea il file RTF con il primo campo "0,Fire"
                        with open(rtf_path, "w") as f:
                            f.write('0,Smoke')
                    video_dest_pathname = os.path.join(dest_folder_path+"/VIDEOS/TRAINING_SET/1", video_name)
                    shutil.copy(video_pathname, video_dest_pathname)

def convert_avi_to_mp4(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".avi"):
                avi_path = os.path.join(root, file)
                mp4_path = os.path.splitext(avi_path)[0] + ".mp4"
                stream = ffmpeg.input(avi_path)
                stream = ffmpeg.output(stream, mp4_path)
                ffmpeg.run(stream)
                os.remove(avi_path)