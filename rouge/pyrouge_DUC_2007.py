from pyrouge import Rouge155
import time
from definitions import ROOT_DIR
start_time = time.time()

if __name__ == "__main__":
    root_directory = ROOT_DIR
    rouge_dir = root_directory + '/rouge/ROUGE-1.5.5'

    system_folder = "test"

    rouge_args = '-e ROUGE-1.5.5/data -n 2 -m -2 4 -l 250 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a'
    # '-e', self._rouge_data,                           # '-a',  # evaluate all systems
    # '-n', 4,  # max-ngram                             # '-x',  # do not calculate ROUGE-L
    # '-2', 4,  # max-gap-length                        # '-u',  # include unigram in skip-bigram
    # '-c', 95,  # confidence interval                  # '-r', 1000,  # number-of-samples (for resampling)
    # '-f', 'A',  # scoring formula                     # '-p', 0.5,  # 0 <= alpha <=1
    # '-t', 0,  # count by token instead of sentence    # '-d',  # print per evaluation scores

    rouge = Rouge155(rouge_dir, rouge_args)

    rouge.model_dir = root_directory + '/Data/DUC_2007/Human_Summaries'
    rouge.system_dir = root_directory + '/Data/DUC_2007/' + system_folder

    rouge.model_filename_pattern = 'summary_#ID#.[A-Z].1.txt'
    rouge.system_filename_pattern = 'D07(\d+)[A-Z].(\w+)'

    rouge_output = rouge.convert_and_evaluate()

    print (rouge_output)
    print("Execution time: " + str(time.time() - start_time))