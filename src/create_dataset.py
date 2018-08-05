import codecs
import csv
import os
from random import shuffle

dataset_path = "dataset/labels.csv"

consonants = []
vowels = []
numerals = []

csv_list = [""]


def create_csv(csv_filepath, position=0):
    with open(csv_filepath, 'rt', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            if line[0] == '' or line[0] == "Class":
                continue
            elif line[0] == "Numerals":
                position = 1
                continue
            elif line[0] == "Vowels":
                position = 2
                continue
            elif line[0] == "Consonants":
                position = 3
                continue

            if position == 1:
                path = "dataset/numerals/" + line[0] + "/"
                for file in os.listdir(path):
                    csv_list.append(path + file.title() + "," + line[2])
            elif position == 2:
                path = "dataset/vowels/" + line[0] + "/"
                for file in os.listdir(path):
                    csv_list.append(path + file.title() + "," + line[2])
            elif position == 3:
                path = "dataset/consonants/" + line[0] + "/"
                for file in os.listdir(path):
                    csv_list.append(path + file.title() + "," + line[2])


def save_data(data):
    file = codecs.open("dataset_character.csv", "w+","utf-8")
    for str in data:
        print(str)
        file.write(str)
        file.write("\n")


create_csv(dataset_path)
shuffle(csv_list)
save_data(csv_list)
