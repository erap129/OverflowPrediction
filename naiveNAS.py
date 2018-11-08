from models_generation import random_model, finalize_model, mutate_net, target_model, set_target_model_filters,\
    genetic_filter_experiment_model, breed, get_evolution_model_filters
import pandas as pd
import numpy as np
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_models import convert_to_dilated
import os
import keras.backend as K
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
import queue
import datetime
import pickle
import treelib
import gp
WARNING = '\033[93m'
ENDC = '\033[0m'


class Tree(object):
    def __init__(self):
        self.children = []
        self.data = None


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def delete_from_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

class NaiveNAS:
    def __init__(self, n_classes, input_time_len, n_chans,
                 X_train, y_train, X_valid, y_valid, X_test, y_test, subject_id, cropping=False):
        self.n_classes = n_classes
        self.n_chans = n_chans
        self.input_time_len = input_time_len
        self.X_train = X_train
        self.X_test = X_test
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_test = y_test
        self.y_valid = y_valid
        self.finalize_flag = 0
        self.cropping = cropping
        self.subject_id = subject_id

    def evaluate_model(self, model):
        K.clear_session()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        finalized_model = finalize_model(model.layer_collection, naive_nas=self)
        finalized_model.model.summary()
        if self.cropping:
            finalized_model.model = convert_to_dilated(model.model)
        best_model_name = 'keras_models/best_keras_model' + str(time.time()) + '.hdf5'
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5)
        mcp = ModelCheckpoint(best_model_name, save_best_only=True, monitor='val_acc', mode='max')
        start = time.time()
        r = finalized_model.model.fit(self.X_train, self.y_train, epochs=50,
                                  validation_data=(self.X_valid, self.y_valid),
                                  callbacks=[earlystopping, mcp], verbose=1)
        n_epochs = len(r.history['loss'])
        finalized_model.model.load_weights(best_model_name)
        res_test = finalized_model.model.evaluate(self.X_test, self.y_test)[1] * 100
        res_train = finalized_model.model.evaluate(self.X_train, self.y_train)[1] * 100
        res_val = finalized_model.model.evaluate(self.X_valid, self.y_valid)[1] * 100
        end = time.time()
        final_time = end-start
        delete_from_folder('keras_models')
        return final_time, n_epochs, res_test, res_val, res_train, finalized_model

    def find_best_model_evolution(self):
        nb_evolution_steps = 10
        tournament = \
            gp.TournamentOptimizer(
                population_sz=10,
                init_fn=random_model,
                mutate_fn=mutate_net,
                naive_nas=self)

        for i in range(nb_evolution_steps):
            print('\nEvolution step:{}'.format(i))
            print('================')
            top_model = tournament.step()
            # keep track of the experiment results & corresponding architectures
            name = "tourney_{}".format(i)
            res, _ = self.evaluate_model(top_model, test=True)
        return res

    def find_best_model_bb(self, folder_name, time_limit=12 * 60 * 60):
        curr_models = queue.Queue()
        operations = [self.factor_filters, self.add_conv_maxpool_block]
        initial_model = self.base_model()
        curr_models.put(initial_model)
        res = self.evaluate_model(initial_model)
        total_results = []
        result_tree = treelib.Tree()
        result_tree.create_node(data=res, identifier=initial_model.name)
        start_time = time.time()
        while not curr_models.empty() and time.time() - start_time < time_limit:
            curr_model = curr_models.get_nowait()
            for op in operations:
                model = op(curr_model)
                res = self.evaluate_model(model)
                print('node name is:', model.name)
                result_tree.create_node(data=res, identifier=model.name, parent=curr_model.name)
                result_tree.show()
                print('model accuracy:', res[1] * 100)
                total_results.append(res[1])
                curr_models.put(model)
        print(total_results)
        return total_results.max()

    def find_best_model_simanneal(self, folder_name, experiment=None, time_limit=12 * 60 * 60):
        curr_model = self.base_model()
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5)
        mcp = ModelCheckpoint('best_keras_model.hdf5', save_best_only=True, monitor='val_acc', mode='max')
        start_time = time.time()
        curr_acc = 0
        num_of_ops = 0
        temperature = 10
        coolingRate = 0.003
        operations = [self.factor_filters, self.add_conv_maxpool_block]
        total_time = 0
        if experiment == 'filter_experiment':
            results = pd.DataFrame(columns=['conv1 filters', 'conv2 filters', 'conv3 filters',
                                            'accuracy', 'runtime', 'switch probability', 'temperature'])
        while time.time()-start_time < time_limit and not self.finalize_flag:
            K.clear_session()
            op_index = random.randint(0, len(operations) - 1)
            num_of_ops += 1
            mcp = ModelCheckpoint('keras_models/best_keras_model' + str(num_of_ops) + '.hdf5',
                save_best_only=True, monitor='val_acc', mode='max', save_weights_only=True)
            model = operations[op_index](curr_model)
            finalized_model = self.finalize_model(model.layer_collection)
            if self.cropping:
                finalized_model.model = convert_to_dilated(model.model)
            start = time.time()
            finalized_model.model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                         callbacks=[earlystopping, mcp])
            finalized_model.model.load_weights('keras_models/best_keras_model' + str(num_of_ops) + '.hdf5')
            end = time.time()
            total_time += end - start
            res = finalized_model.model.evaluate(self.X_test, self.y_test)[1] * 100
            curr_model = model
            if res >= curr_acc:
                curr_model = model
                curr_acc = res
                probability = -1
            else:
                probability = np.exp((res - curr_acc) / temperature)
                rand = np.random.choice(a=[1, 0], p=[probability, 1-probability])
                if rand == 1:
                    curr_model = model
            temperature *= (1-coolingRate)
            print('train time in seconds:', end-start)
            print('model accuracy:', res * 100)
            if experiment == 'filter_experiment':
                results.loc[num_of_ops - 1] = np.array([int(finalized_model.model.get_layer('conv1').filters),
                                                    int(finalized_model.model.get_layer('conv2').filters),
                                                    int(finalized_model.model.get_layer('conv3').filters),
                                                    res,
                                                    str(end-start),
                                                    str(probability),
                                                    str(temperature)])
                print(results)
        final_model = self.finalize_model(curr_model.layer_collection)
        final_model.model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                        callbacks=[earlystopping])
        res = final_model.model.evaluate(self.X_test, self.y_test)[1] * 100
        print('model accuracy:', res)
        print('average train time per model', total_time / num_of_ops)
        now = str(datetime.datetime.now()).replace(":", "-")
        if experiment == 'filter_experiment':
            if not os.path.isdir('results/'+folder_name):
                createFolder('results/'+folder_name)
            results.to_csv('results/'+folder_name+'/subject_' + str(self.subject_id) + experiment + '_' + now + '.csv', mode='a')

    def evolution_filters(self):
        pop_size = 10
        select_fittest = 2
        select_lucky_few = 2
        num_generations = 100
        population = []
        breed_rate = 0.8
        mutation_rate = 0.3
        results = pd.DataFrame(columns=['best model acc', 'best model time', 'best model epoch num', 'best model filters',
                                        'mean model acc', 'mean model time', 'mean model epoch num'])
        for i in range(pop_size):
            population.append(genetic_filter_experiment_model(num_blocks=5))
        for generation in range(num_generations):
            weighted_population = []
            for i, pop in enumerate(population):
                final_time, n_epochs, _, res_val, _, _ = self.evaluate_model(pop)
                weighted_population.append((pop, res_val, final_time, n_epochs))
            weighted_population = sorted(weighted_population, key=lambda x: x[1], reverse=True)
            print('fittest individual in generation', generation, 'has fitness:', weighted_population[0][1])
            print('mean fitness of population is', np.mean([weight for (model, weight, final_time, n_epochs) in weighted_population]))
            best_model_filters = get_evolution_model_filters(weighted_population[0][0])
            results.loc[generation] = np.array([weighted_population[0][1],
                                                weighted_population[0][2],
                                                weighted_population[0][3],
                                                best_model_filters,
                                                np.mean([weight for (model, weight, final_time, n_epochs) in weighted_population]),
                                                np.mean([final_time for (model, weight, final_time, n_epochs) in weighted_population]),
                                                np.mean([n_epochs for (model, weight, final_time, n_epochs) in weighted_population])])
            fittest = weighted_population[0:select_fittest]
            lucky_few_indexes = random.sample(range(select_fittest+1, len(population)), k=select_lucky_few)
            lucky_few = [weighted_population[i] for i in lucky_few_indexes]
            to_breed = fittest + lucky_few
            np.random.shuffle(to_breed)
            new_population = []
            for i in range(int(len(to_breed) / 2)):
                for j in range(int(num_generations / (len(to_breed) / 2))):
                    new_population.append(breed(first_model=to_breed[i][0],
                                second_model=to_breed[len(to_breed) - 1 - i][0], mutation_rate=mutation_rate, breed_rate=breed_rate))
            print(results)
        now = str(datetime.datetime.now()).replace(":", "-")
        results.to_csv('results/filter_size_evolution_' + 'pop_' + str(pop_size) +
                       '_gen_' + str(num_generations) + '_' + now + '.csv', mode='a')

    def grid_search_filters(self, lo, hi, jumps):
        model = target_model()
        start_time = time.time()
        num_of_ops = 0
        total_time = 0
        results = pd.DataFrame(columns=['conv1 filters', 'conv2 filters', 'conv3 filters',
                                         'test acc', 'val acc', 'train acc', 'runtime'])
        for first_filt in range(lo, hi, jumps):
            for second_filt in range(lo, hi, jumps):
                for third_filt in range(lo, hi, jumps):
                    K.clear_session()
                    num_of_ops += 1
                    model = set_target_model_filters(model, first_filt, second_filt, third_filt)
                    run_time, res_test, res_val, res_train, _ = self.evaluate_model(model)
                    total_time += run_time
                    print('train time in seconds:', run_time)
                    print('model accuracy:', res_test)
                    results.loc[num_of_ops - 1] = np.array([int(model.model.get_layer('conv1').filters),
                                                            int(model.model.get_layer('conv2').filters),
                                                            int(model.model.get_layer('conv3').filters),
                                                            res_test,
                                                            res_val,
                                                            res_train,
                                                            str(run_time)])
                    print(results)

        total_time = time.time() - start_time
        print('average train time per model', total_time / num_of_ops)
        now = str(datetime.datetime.now()).replace(":", "-")
        results.to_csv('results/filter_gridsearch_' + now + '.csv', mode='a')

    def grid_search_kernel_size(self, lo, hi, jumps):
        model = target_model()
        start_time = time.time()
        num_of_ops = 0
        total_time = 0
        results = pd.DataFrame(columns=['conv1 kernel size', 'conv2 kernel size', 'conv3 kernel size',
                                         'accuracy', 'train acc', 'runtime'])
        for first_size in range(lo, hi, jumps):
            for second_size in range(lo, hi, jumps):
                for third_size in range(lo, hi, jumps):
                    K.clear_session()
                    num_of_ops += 1
                    model = set_target_model_kernel_sizes(model, first_size, second_size, third_size)
                    run_time, res, res_train = self.run_one_model(model)
                    total_time += time
                    print('train time in seconds:', time)
                    print('model accuracy:', res)
                    results.loc[num_of_ops - 1] = np.array([int(model.model.get_layer('conv1').kernel_size[1]),
                                                            int(model.model.get_layer('conv2').kernel_size[1]),
                                                            int(model.model.get_layer('conv3').kernel_size[1]),
                                                            res_train,
                                                            str(run_time)])
                    print(results)

        total_time = time.time() - start_time
        print('average train time per model', total_time / num_of_ops)
        now = str(datetime.datetime.now()).replace(":", "-")
        results.to_csv('results/kernel_size_gridsearch_' + now + '.csv', mode='a')





