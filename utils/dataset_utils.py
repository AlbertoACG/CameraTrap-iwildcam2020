# Author: Fagner Cunha

"""Basic set of tools for dataset operations.

Contains functions for baisc operations such as
filtering, train/test split and others.
"""

from __future__ import print_function
from __future__ import division

from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import pandas as pd

import sys
import random
import os
import uuid
from shutil import copyfile
from .ga_utils import PopulationGenerator

def _percent(value, total):
    """Calculates percentage value.
    
    # Arguments
        value: value for which the percentage will be calculated.
        total: value corresponding to 100%.
    
    # Returns
        The percentage.
    
    """
    return 100*value/total

def group_split(groups,
                splits = {'train': 0.7, 'validation': 0.15, 'test': 0.15},
                print_results = True,
                seed = {}):
    """Performs a random split of a dataset based on groups of instances.
    
    # Arguments
        groups: A dictionary containing the number of images for each group.
        splits: A dictionary containing the percentage for each dataset split.
        print_results: Print a summary of split performed.
        seed: A dictionary with a pre-selection for specific groups
        
    # Returns
        groups_split: a dictionary containing the split for each group.
        split_list: a dictionary containing a list of groups for each split.
    
    """
    sets_names = splits.keys()
    num_imgs = {k: 0 for k, v in splits.items()}
    split_list = {k: [] for k, v in splits.items()}
    groups_split = {}
    total_images = sum(groups.values())
    
    groups_keys = list(groups.keys())
    
    for group, split_selected in seed.iteritems():
        groups_keys.remove(group)
        num_imgs[split_selected] += groups[group]
        groups_split[group] = split_selected
        split_list[split_selected] += [group]
        
        if (num_imgs[split_selected] / total_images) > splits[split_selected]:
            sets_names.remove(split_selected)
    
    #make sure the dictonary key order does not affect the random split
    random.shuffle(groups_keys)
    
    #randomly distribuites the groutps among sets util complete its percentage
    for group in groups_keys:
        split_selected = random.choice(sets_names)
        num_imgs[split_selected] += groups[group]
        groups_split[group] = split_selected
        split_list[split_selected] += [group]
        
        if (num_imgs[split_selected] / total_images) > splits[split_selected]:
            sets_names.remove(split_selected)
    
    if print_results:
        print("Dataset split:\n")
        
        for split in splits.keys():
            print("%s: %d (%.2f%%)" % (split.title(), num_imgs[split],
                                       _percent(num_imgs[split], total_images)))
        print("Total: %d" % (total_images))
        
    return groups_split, split_list
    
def calculate_mse_per_split(data, columns, target_value):
    """Calculates the mean squared error of values in the columns specified.
    
    #Arguments
        data: a pandas dataframe containing the data to be evaluated.
        columns: a list of columns of data to be evaluated.
        target_value: the expected value.
    
    #Returns
        The mean squared error of all values in the columns specified.
    
    """
    values = []
    for column in columns:
        values = values + list(data[column])
        
    values = np.array(values)
    mse = ((target_value - values) **2).mean()
    
    return mse

def is_valid_split(data, columns, min_value):
    """Verifies if a split is valid, i.e., all values in the specified columns
        have at least a minimum value.
    
    #Arguments
        data: a pandas dataframe containing the data to be evaluated.
        columns: a list of columns of data to be evaluated.
        min_value: minimum value.
    
    """
    for column in columns:
        for value in list(data[column]):
            if value < min_value:
                return False
    
    return True

def valid_group_split(instaces, groups, columns, min_value,
                       used_classes = None,
                       **kwargs):
    """Performs a valid random split of a dataset based on groups of instances.
    
    #Arguments
        instances: a pandas dataframe (columns = ['instanceId', 'classId',
            'groupId']) containing a list of all instances.
        groups: A dictionary containing the number of images for each group.
        columns: a list of split columns to be validated.
        min_value: minimum value that a colmun can have to be valid.
        used_classes: a list of classes that will be used to verify if the split
            is valid. The name used are the same defined by classId in the
             argument 'instances'
       
       The ramining arguments are the same as for the group_split function.
       
    #Returns
        groups_split: a dictionary containing a split for each group.
        split_list: a dictionary containing a list of groups for each split.
    
    """
    
    while True:
        split, split_list = group_split(groups, **kwargs)
        data = calculate_splits_per_class(instaces, split)
        if used_classes is not None:
            data = data.loc[used_classes]
        
        if(is_valid_split(data, columns, min_value)):
            break
    
    return split, split_list

def _read_image(image_file, target_size):
    """Read a image from file and returns as NumPy array.
    
    # Arguments
        image_file: image file name.
        target_size: target size of the image.
        
    # Returns
        A NumPy array of the image read.
    """
    
    img = load_img(image_file, target_size = target_size)
    img = img_to_array(img)
    return img

def _get_zero_image(dummy_image, target_size):
    """Creates a zero NumPy array with a target size image shape
    
    # Arguments
        dummy_image: dummy image to be read and used as shape.
        target_size: target size of the image.
    
    # Returns
        A NumPy array of zeros as the shape of dummy image.
    """

    img = _read_image(dummy_image, target_size)
    return np.zeros(img.shape)

def calculate_mean_image_from_files(files_lst, target_size):
    """Calculate the mean image for a files list.
    
    # Arguments
        files_lst: list of images files to calculate mean
        target_size: target size of the image.
        
    #Returns
        The mean image for the files
        
    """
    
    mean_img = _get_zero_image(files_lst[0], target_size)
    
    for img_file in files_lst:
        mean_img += _read_image(img_file, target_size)
    
    mean_img = mean_img / len(files_lst)
    
    return mean_img

def calculate_std_image_from_files(files_lst, target_size, mean_img):
    """Calculate the standard deviation image for a files list.
    
    # Arguments
        files_lst: list of images files to calculate standard deviation 
        target_size: target size of the image.
        
    #Returns
        The standard deviation image for the files
        
    """
    std_img = _get_zero_image(files_lst[0], target_size)
    
    for img_file in files_lst:
        img = _read_image(img_file, target_size)
        std_img += np.power(img - mean_img, 2)
    
    std_img = np.sqrt(std_img / len(files_lst))
    
    return std_img

def calculate_splits_per_class(instances, class_split):
    """Calculates the percentage split for each class

    # Arguments
        instances: a pandas dataframe (columns = ['instanceId', 'classId', 'groupId']) containing a list of all instances.
        class_split: a dictionary containing the split for each group.

    # Returns
        A pandas dataframe containing the quantities and percents splits for each class

    """

    class_split = pd.DataFrame(class_split.items(), columns=['groupId', 'split'])
    data = pd.merge(instances, class_split, on='groupId', how='left')
    
    datasets_distribution = pd.DataFrame()
    datasets_distribution['Classes'] = list(instances.classId.unique())
    datasets_distribution = datasets_distribution.set_index('Classes')
    
    for split in list(class_split.split.unique()):
        datasets_distribution[split] = data[data['split'] == split].classId.value_counts()
    datasets_distribution.fillna(0, inplace=True)
    
    percents = datasets_distribution.div(datasets_distribution.sum(axis=1), axis=0).mul(100)
    percents.columns = [split + ' %' for split in list(class_split.split.unique())]

    return pd.concat([datasets_distribution, percents], axis=1)

def list_all_pic_from_directory_recursively(basedir, white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}):
    """List all pictures from a directory recursively
    
    # Arguments
        basedir: directory base for the search
        white_list_formats: a list of allowed file extensions
    
    # Returns
        A list of all files
    
    """
    
    all_files = []

    for root, directories, filenames in os.walk(basedir):
        for filename in filenames:
            fname = os.path.relpath(os.path.join(root,filename), basedir)
            
            if white_list_formats is not None:
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    all_files.append(fname)
            else:
                all_files.append(fname)
    
    return all_files

def resize_images(originDir, targetDir, target_size = (450, 450), overwrite = True):
    """Resize all images from a directory to a specific size
    
    # Arguments
        originDir: directory containing original images
        targetDir: ditectory where the resized images will be saved
        target_size: target size of images
        overwrite: whether it should overwrite existing images with same name
    
    """
    
    files = list_all_pic_from_directory_recursively(originDir)
    
    for fname in files:
        new_file = os.path.join(targetDir, fname)
        
        if(overwrite or not os.path.exists(new_file)):
            try:
                img = Image.open(os.path.join(originDir, fname))
                img = img.resize(target_size, Image.ANTIALIAS)
            except IOError:
                print ("error: ", fname)
                sys.stdout.flush()
                continue
            
            if not os.path.exists(os.path.dirname(new_file)):
                os.makedirs(os.path.dirname(new_file))
            img.save(new_file)

def create_dataset_directory(instances, group_split, origin_path, target_path, ignore_partitions=[]):
    """Creates the dataset directory structure required by Keras ImageDataGenerator
    
    # Arguments
        instances: A list of instances; each instance is a list containing 3 elements: image path, group id, class
        group_split: A dictionary containing the split for each group
        origin_path: Path to the original images
        target_path: Path where dataset struct will be created
        ignore_partitions: List of partitions to be ignored
    
    # Returns
        A list of the files structure. Each image will have a new name based on an uuid
    
    """
    
    data_out = []
    for instance in instances:
        if group_split[instance[1]] not in ignore_partitions:
            origin_file = os.path.join(origin_path, instance[0])

            _, target_file = os.path.splitext(origin_file)
            target_file = str(uuid.uuid4()) + target_file
            target_file = os.path.join(instance[2], target_file)
            target_file = os.path.join(group_split[instance[1]], target_file)
            rel_target_file = target_file
            target_file = os.path.join(target_path, target_file)

            if os.path.isfile(origin_file):
                if not os.path.exists(os.path.dirname(target_file)):
                    os.makedirs(os.path.dirname(target_file))
                copyfile(origin_file, target_file)
                data_out = data_out + [[instance[0], rel_target_file]]

    return data_out

class GroupSplitPopulationGenerator(PopulationGenerator):
    """Provide data to the GeneticAlgorithmRunner based on group split problem.
    
    #Arguments
        instances: a pandas dataframe (columns = ['instanceId', 'classId',
            'groupId']) containing a list of all instances.
        groups: A dictionary containing the number of images for each group.
        partitions: A dictionary containing the percentage for each partition.
        target_columns: a list of split columns to be validated.
        min_target_value: minimum value that a colmun can have to be valid.
        target_value: the expected value of validated columns.
        used_classes: a list of classes that will be used to verify if the split
            is valid. The name used are the same defined by classId in the
            argument 'instances'
        max_retry: maximum number of tentatives to retry operations when got 
            invalid results
        
    """
    
    def __init__(self,
                instances,
                groups,
                partitions = {'train': 0.7, 'validation': 0.15, 'test': 0.15},
                target_columns = ['validation %', 'test %'],
                min_target_value = 0,
                target_value = 15,
                used_classes = None,
                max_retry = 1000):
        self.instances = instances
        self.groups = groups
        self.partitions = partitions
        self.target_columns = target_columns
        self.min_target_value = min_target_value
        self.target_value = target_value
        self.used_classes = used_classes
        self.max_retry = max_retry
        
    def generateIndividual(self):
        split, split_list = valid_group_split(self.instances, self.groups,
                                self.target_columns, self.min_target_value,
                                used_classes = self.used_classes,
                                splits=self.partitions,
                                print_results=False)
        return {'groups': split, 'partitions': split_list}
    
    def fitnessIndividual(self, individual):
        data = calculate_splits_per_class(self.instances, individual['groups'])
        if self.used_classes is not None:
            data = data.loc[self.used_classes]
        
        return calculate_mse_per_split(data, self.target_columns, self.target_value)
    
    def createChild(self, individual1, individual2):
        
        for i in range(self.max_retry):
            split = {}
            split_list = {x: [] for x in list(set(individual1['groups'].values()))}
            
            for group in individual1['groups'].keys():
                if (int(100 * random.random()) < 50):
                    split[group] = individual1['groups'][group]
                    split_list[individual1['groups'][group]] += [group]
                else:
                    split[group] = individual2['groups'][group]
                    split_list[individual2['groups'][group]] += [group]
            
            data = calculate_splits_per_class(self.instances, split)
            if self.used_classes is not None:
                data = data.loc[self.used_classes]
            
            if(is_valid_split(data, self.target_columns, self.min_target_value)):
                child = {}
                child['groups'] = split
                child['partitions'] = split_list
                
                return child
        
        #If it was not possible create a child, returns a clone
        if (int(100 * random.random()) < 50):
            return individual1
        else:
            return individual2
    
    def mutateIndividual(self, individual):
        
        for i in range(self.max_retry):
            ind = individual.copy()
            
            group = random.choice(ind['groups'].keys())
            partition = random.choice(list(set(ind['groups'].values())))
            
            ind['partitions'][ind['groups'][group]].remove(group)
            ind['groups'][group] = partition
            ind['partitions'][partition] += [group]
            
            data = calculate_splits_per_class(self.instances, ind['groups'])
            if self.used_classes is not None:
                data = data.loc[self.used_classes]
            
            if(is_valid_split(data, self.target_columns, self.min_target_value)):
                return ind
        
        #If it was not possible create a valid mutation child, returns a clone
        return individual
