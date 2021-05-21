from aifeynman import AIFeynmanRegressor

hyper_params = []
for bftt, nne in [[60,4000],[10*60,400]]:
    for ops in ["10ops.txt","14ops.txt","19ops.txt"]:
        hyper_params.append(dict(
            BF_try_time=[bftt],
            NN_epochs=[nne],
            ))

est = AIFeynmanRegressor(
        BF_try_time=60,
        polyfit_deg=4,
        NN_epochs=4000,
        max_time=2*60*60
        tmp_dir='/scratch/aifeynman'
        )

def complexity(est):
    return est.complexity()

def model(est):
    return est.best_model_
