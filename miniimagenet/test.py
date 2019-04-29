#!/usr/bin/python3
import pytest
import task_generator as tg
import argparse

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=10)
parser.add_argument("-e", "--episode", type=int, default=500000)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
args = parser.parse_args()

FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit


def test_task_generator():
    print("this is test about task_generator!!!")
    metatrain_folders, metatest_folders = tg.mini_imagenet_folders()
    task = tg.MiniImagenetTask(metatrain_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
    sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                         shuffle=False)
    batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test",
                                                        shuffle=True)


if __name__ == '__main__':
    test_task_generator()
