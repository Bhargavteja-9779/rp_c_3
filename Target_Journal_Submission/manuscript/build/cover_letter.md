[Corresponding author name]\
[Department, Institution]\
[Address, City, Country]\
[E-mail] · ORCID [0000-0000-0000-0000]

[Date]

The Editor-in-Chief\
*{{JNAME}}*\
{{PUBLISHER}}

**Submission of an original research article**

Dear Editor-in-Chief,

We submit the manuscript entitled **"Knowing when to ask a human: reliable confidence for fast time series classifiers under sensor faults"** for consideration as an original research article in *{{JNAME}}*.

**Summary.** Many monitoring systems, from wearable heart monitors to industrial sensors, let a classifier decide on its own and pass uncertain cases to a person. This only works if the classifier's confidence drops when a recording is damaged, for example by electrode noise, drift, spikes or dropouts. Fast random convolutional kernel classifiers such as MiniRocket are accurate and cheap enough for such systems, but their confidence does not drop on faulty recordings. We propose LOOM, a closed-form readout that scales each prediction by how far the recording lies from the training data, measured by the ridge leverage, and that is calibrated on exact leave-one-out quantities without held-out data, refitting or ensembles. On {{S['n_datasets']}} UCR datasets with a pre-registered benchmark of 18 simulated sensor faults, and in a human-in-the-loop study on cardiac, wearable and industrial sensor data, {{COVER_RESULT}}

**Fit with the journal.** {{FIT}} It offers a practical, computationally light solution for monitoring applications in health care and industry, where data are scarce and decisions are shared between a model and a human reviewer.

**Originality and relation to prior work.** The closed-form leave-one-out calibration of ridge readouts was introduced by Dempster et al. (PreVal, arXiv:2401.15610). We cite it as the closest prior work and use it as the primary baseline. Our contribution is the per-input leverage moderation fitted on held-out leverages, the closed-form jackknife+ sets, and the evaluation under distribution shift. Negative results are reported: a small in-distribution effect, a higher in-distribution ECE caused by target smoothing, no benefit for shifts that leave the random-kernel features in the span of the training data, and lower robustness than the tree-based QUANT classifier under the strongest corruptions.

**Declarations.** This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have approved the submission. The study uses only public benchmark data; no ethics approval was required. Code and all machine-generated results will be made publicly available; an anonymised link is provided for review. The use of an AI assistant is declared in the manuscript, as the journal's policy requires. [Funding and competing interests: to be completed by the authors.]

We confirm that the manuscript has been prepared for double-anonymous review, with author information provided on a separate title page.

Thank you for considering our manuscript.

Yours sincerely,

[Corresponding author name], on behalf of all authors
