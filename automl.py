import autogluon as ag
from autogluon import ImageClassification as task

dataset = task.Dataset('train')
test_dataset = task.Dataset('va', train=False)
print(ag.get_gpu_count())

time_limits = 8*60*60
epochs = 150
output = 'checkpoint/'
resume = True

classifier = task.fit(dataset,
                      search_strategy='skopt',
                      search_options={'base_estimator': 'RF', 'acq_func': 'EI'},
                      time_limits=time_limits,
                      epochs=epochs,
                      ngpus_per_trial=1,
                      num_trials=1,
                      output_directory=output,
                      verbose=True,
                      resume=resume,
                      plot_results=True)


print('Top-1 val acc: %.3f' % classifier.results[classifier.results['reward_attr']])
classifier.save('checkpoint/model.pth')

