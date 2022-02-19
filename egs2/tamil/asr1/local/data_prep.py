# Copyright 2021  Young Min Kim
#           2021  Carnegie Mellon University
# Apache 2.0

import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


if len(sys.argv) != 2:
    print("Usage: python data_prep.py [DATASET_DIR]")
    sys.exit(1)


DATASET_DIR = sys.argv[1]  # raw dataset zip file contents should be here
DATA_DIR = "data"  # processed data should go here


def read_tamil_data(audio_csv, sentences_csv, export_csv):
    sent_df = pd.read_csv(sentences_csv)
    audio_df = pd.read_csv(audio_csv)
    output_df = []

    # if not os.path.exists("wavs"):
    #     os.mkdir("wavs")

    for i in range(len(sent_df)):
        intent, intent_details, inflection, transcript = (
            sent_df.iloc[i]["intent"],
            sent_df.iloc[i]["intent_details"],
            sent_df.iloc[i]["inflection"],
            sent_df.iloc[i]["sentence"],
        )

        for j in range(len(audio_df)):
            wav_name, intent_, inflection_ = (
                audio_df.iloc[j]["audio_file"],
                audio_df.iloc[j]["intent"],
                audio_df.iloc[j]["inflection"],
            )
            if intent_ == intent and inflection_ == inflection:
                # clean transcript
                # export audio of for the crop with wav_path_start_duration
                export_path = os.path.join(DATASET_DIR, "audio_files", wav_name)
                # Append to output_df
                output_df.append(
                    [
                        export_path,
                        "unknown",
                        transcript,
                        intent_details.replace(" ", ""),
                    ]
                )

    X = pd.DataFrame(
        output_df, columns=["path", "speakerId", "transcription", "task_type"]
    )
    Y = X.pop("task_type").to_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, stratify=Y, test_size=0.20, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, stratify=y_test, test_size=0.50, random_state=42
    )

    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(DATA_DIR, "train.csv"))
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(DATA_DIR, "test.csv"))
    pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(DATA_DIR, "validation.csv"))


read_tamil_data(
    os.path.join(DATASET_DIR, "Tamil_Data.csv"),
    os.path.join(DATASET_DIR, "Tamil_Sentences.csv"),
    os.path.join(DATASET_DIR, "export.csv")
)


dir_dict = {
    "train": "train.csv",
    "valid": "validation.csv",
    "test": "test.csv",
}


for partition in dir_dict:
    with open(os.path.join(DATA_DIR, partition, "text"), "w") as text_f, \
         open(os.path.join(DATA_DIR, partition, "wav.scp"), "w") as wav_scp_f, \
         open(os.path.join(DATA_DIR, partition, "transcript"), "w") as transcript_f, \
         open(os.path.join(DATA_DIR, partition, "utt2spk"), "w") as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        transcript_df = pd.read_csv(os.path.join(DATA_DIR, dir_dict[partition]))
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            words = row[4].replace(" ", "_") + " " + " ".join([ch for ch in row[3]])
            path_arr = row[1].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + row[1] + "\n")
            utt2spk_f.write(utt_id + " " + row[2] + "\n")