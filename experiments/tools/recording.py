import datetime
import json

import os


def record_shared(progress, run_start, epoch, samples_trained):
    progress.update({
        'seconds_elapsed': (datetime.datetime.now() - run_start).seconds,
        'sample_exposure': samples_trained,
        'epoch': epoch,
    })


def record_epoch(progress_array, run_start, epoch, samples_trained, f_network, test_accuracy, test_loss, train_accuracy, train_loss):
    print("Train: {:.1f}% @ L={:.3f}".format(train_accuracy, train_loss))
    print("Test:  {:.1f}% @ L={:.3f}".format(test_accuracy, test_loss))
    progress = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'singular_values': [values.tolist() for values in f_network.singular_value_sets()],
    }
    record_shared(progress, run_start, epoch, samples_trained)
    progress_array.append(progress)

    return train_loss, test_loss


def record_batch(progress_array, run_start, batches_completed, epoch, samples_trained, batch_accuracy, batch_loss):
    progress = {
        'batches_completed': batches_completed,
        'batch_accuracy': batch_accuracy,
        'bach_loss': batch_loss,
    }
    record_shared(progress, run_start, epoch, samples_trained)
    progress_array.append(progress)


def save_run_data(db, out_path, run_name, run_start, run_number):
    os.makedirs(out_path, exist_ok=True)
    run_timestamp = run_start.isoformat()
    out_filename = "{} - {} - run {}.json".format(run_timestamp, run_name, run_number)
    with open(os.path.join(out_path, out_filename), 'w') as file:
        file.write(json.dumps(db, indent=2))