from slime.utils.types import Sample
from slime.rollout.sglang_rollout import generate

async def custom_generate(args, sample: Sample, sampling_params:dict) -> Sample | list[Sample]:
    teacher_sample = await generate(args, sample, sampling_params)

    if teacher_sample.status == Sample.Status.ABORTED:
        return teacher_sample

    #change the prompt for the test sample to have reduced context and create a new sample

    

    test_sample = await generate(args, sample, sampling_params)
    #take teacher log probs and shove them in
    test_sample.teacher_log_probs = teacher_sample.rollout_log_probs
    return test_sample
    