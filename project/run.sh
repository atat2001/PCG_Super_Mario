
#    parser = argparse.ArgumentParser(description='Running a Generato from the list of generators. You can adjust any hyperparameter of any algorithm based on their extra inputs. For example, GA/ES/Random can set the fitness function by --fitness followed by name of the function in the search.py file.')
#    parser.add_argument('outputfolder', type=str, help='the folder to save the search')
#    parser.add_argument('-p', '--problem', type=str, default='binary-v0', help='what is the problem that you are trying to solve (`binary-v0`, `zelda-v0`, `sokoban-v0`, ...).')
#    parser.add_argument('-g', '--generator', type=str, default='random', help='what is the generator file name from generators folder (`ga`, `es`, `random`, ...).')
#    parser.add_argument('-s', '--steps', type=int, default=100, help='the number of iterations to run that generator')
#    parser.add_argument('-e', '--early_stop', action="store_true", help="stop generation when the best function is 1")

## right now set with no early stopping: --early_stop
# to set seed: --seed ...
cd "$(dirname "$0")"
python run.py out --problem smb-small-v0 -g genetic -s 100