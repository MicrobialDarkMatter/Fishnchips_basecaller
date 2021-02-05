import re
import math
import datetime
import numpy as np

from src.controllers.file_controller import FileController

class EvaluationController():
    def __init__(self, experiment_name):
        self.file_controller = FileController(experiment_name)

    def count_unmatched_reads(self):
        data = self.get_accuracy_list(include_unmatched=True)
        count = 0
        for accuracy in data:
            if accuracy == 0:
                count += 1
        return count

    def count_unmatched_reads_per_bacteria(self):
        data = self.get_accuracy_list_per_bacteria(include_unmatched=True)
        counts = {}
        for bacteria in data.keys():
            count = 0
            for accuracy in data[bacteria]:
                if accuracy == 0:
                    count += 1
            counts[bacteria] = count
        return counts

    def count_reads(self):
        data = self.file_controller.load_testing()
        return len(data)

    def count_reads_per_bacteria(self):
        data = self.get_accuracy_list_per_bacteria(include_unmatched=True)
        counts = {}
        for bacteria in data.keys():
            counts[bacteria] = len(data[bacteria])
        return counts

    def get_accuracy_list(self, include_unmatched=True):
        data = self.file_controller.load_testing()
        accuracy = []
        for measurement in data:
            if include_unmatched == False and measurement['cigacc'] == 0:
                continue
            accuracy.append(measurement['cigacc'] * 100)
        return accuracy

    def get_accuracy_list_per_bacteria(self, include_unmatched=True):
        data = self.file_controller.load_testing()
        accuracies = {}
        for measurement in data:
            key = measurement['bacteria']
            if key not in accuracies.keys():
                accuracies[key] = []
            accuracies[key].append(measurement['cigacc'] * 100)
        return accuracies

    def get_accuracy_mean(self, include_unmatched=True):
        data = self.get_accuracy_list(include_unmatched)
        data = np.array(data)
        return data.mean()

    def get_accuracy_mean_per_bacteria(self, include_unmatched=True):
        data = self.get_accuracy_list_per_bacteria(include_unmatched)
        means = {}
        for bacteria in data.keys():
            bacteria_data = data[bacteria]
            bacteria_data = np.array(bacteria_data)
            means[bacteria] = bacteria_data.mean()
        return means

    def get_total_testing_time(self):
        total_time_seconds = 0
        data = self.file_controller.load_testing()
        for measurement in data:
            total_time_seconds += measurement['time']
        total_time_seconds = math.floor(total_time_seconds)
        total_time = str(datetime.timedelta(seconds=total_time_seconds))
        return total_time

    def get_total_training_time(self):
        data = self.file_controller.load_training()
        data = np.array(data)
        training_times = data[:,3]
        _, training_stop_idx = self.get_best_validation_loss()
        total_time_seconds = training_times[training_stop_idx] - training_times[0]
        total_time_seconds = math.floor(total_time_seconds)
        total_time = str(datetime.timedelta(seconds=total_time_seconds))
        return total_time

    def get_best_validation_loss(self):
        data = self.file_controller.load_training()        
        data = np.array(data)
        validation_losses = data[:,2]
        min_validation_idx = -1
        min_validation_loss = 1e10
        for i,validation_loss in enumerate(validation_losses):
            if validation_loss < 0:
                continue
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                min_validation_idx = i
        return min_validation_loss, min_validation_idx

    def get_SMDI(self):
        data = self.file_controller.load_testing()
        smdi_dict = {"S":0,"M":0,"D":0,"I":0}
        total_length = 0
        for measurement in data:
            if measurement['cigacc'] == 0:
                continue
            total_length += measurement['blen']
            cigar_string = measurement['cig']
            result = re.findall(r'[\d]+[SMDI]', cigar_string) #[6M, 5D, ...]
            for r in result:
                amount = int(r[:-1]) # 6
                key = r[-1] # M
                smdi_dict[key] += amount
        for key in 'SMDI':
            smdi_dict[key] /= total_length 
        return smdi_dict

    def get_SMDI_per_bacteria(self):
        data = self.file_controller.load_testing()
        smdi_dicts = {}
        lenghts = {}
        for measurement in data:
            if measurement['cigacc'] == 0:
                continue
            bacteria = measurement['bacteria']
            if bacteria not in smdi_dicts.keys():
                smdi_dicts[bacteria] = {"S":0,"M":0,"D":0,"I":0}
                lenghts[bacteria] = 0
            lenghts[bacteria] += measurement['blen']
            cigar_string = measurement['cig']
            result = re.findall(r'[\d]+[SMDI]', cigar_string) 
            for r in result:
                amount = int(r[:-1])
                key = r[-1]
                smdi_dicts[bacteria][key] += amount
        for bacteria in smdi_dicts.keys():
            for key in 'SMDI':
                smdi_dicts[bacteria][key] /= lenghts[bacteria]
        return smdi_dicts