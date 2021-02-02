import time
import traceback 
import mappy as mp

from src.utils.base_converter import convert_to_base_strings
from src.utils.assembler import assemble_and_output
from src.controllers.file_controller import FileController
from src.controllers.inference_controller import InferenceController

class TestingController():
    def __init__(self, config, experiment_name, model, new_testing):
        test_config = config['testing']
        model_config = config['model']
        
        self.model = model
        self.reads = test_config['reads']
        self.batch_size = test_config['batch_size']

        self.use_assembler = test_config['signal_window_stride'] < model_config['signal_window_size']
        self.inference_controller = InferenceController()

        self.file_controller = FileController(experiment_name)
        self.results = [] if new_testing else self.file_controller.load_testing()

    def pretty_print_progress(self, start, end, total):
        progress_str = '['
        for i in range(0, total, total//50):
            if i >= start and i < end:
                progress_str += 'x'
            else:
                progress_str += '-'
        progress_str += ']'
        return progress_str

    def get_assembly(self, y_pred, iteration, read_id, bacteria):
        if self.use_assembler == False:
            return ''.join(y_pred)
        assembly_path = self.file_controller.get_assembly_filepath(iteration, read_id, bacteria)
        return assemble_and_output(assembly_path, y_pred)

    def get_result(self, assembly, aligner, read_id, bacteria):
        try:
            besthit = next(aligner.map(assembly))
            cigacc = 1-(besthit.NM/besthit.blen)
            return self.get_result_dict(read_id, bacteria, besthit.ctg, besthit.r_st, besthit.r_en, besthit.NM, besthit.blen, besthit.cigar_str, cigacc)
        except:
            return self.get_result_dict(read_id, bacteria, 0, 0, 0, 0, 0, 0, 0)

    def get_result_dict(self, read_id, bacteria, ctg, r_st, r_en, nm, blen, cig, cigacc):
        return {
            'read_id':read_id,
            'bacteria':bacteria,
            'ctg': ctg,
            'r_st': r_st,
            'r_en': r_en,
            'NM': nm,
            'blen': blen,
            'cig': cig,
            'cigacc': cigacc
        }

    def test(self, bacteria, generator, aligner):
        print(' - Testing model.')
        for i in range(self.reads):
            try:
                x, read_id = next(generator.get_batched_read())
                start_time = time.time()
                y_pred = []
                for b in range(0, len(x), self.batch_size):
                    x_batch = x[b:b+self.batch_size]
                    print(f"{i:02d}/{self.reads:02d} Predicting windows {self.pretty_print_progress(b, b+len(x_batch), len(x))} {b:04d}-{b+len(x_batch):04d}/{len(x):04d}", end="\r")
                                       
                    y_batch_pred = self.inference_controller.predict_batch(x_batch, self.model)
                    y_batch_pred_strings = convert_to_base_strings(y_batch_pred)
                    y_pred.extend(y_batch_pred_strings)
                
                assembly = self.get_assembly(y_pred, i, read_id, bacteria)
                result = self.get_result(assembly, aligner, read_id, bacteria)
                result['time'] = time.time() - start_time
                self.results.append(result)

                print(f"{i:02d}/{self.reads} Done | CIG ACC: {result['cigacc']}"+" "*70) # 70 blanks to overwrite the previous print
                self.file_controller.save_testing(self.results)
            except Exception as e:
                print(e)
                traceback.print_exc()
