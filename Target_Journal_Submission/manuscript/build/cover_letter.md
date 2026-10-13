[Corresponding author name]
[Department, Institution]
[Address, City, Country]
[E-mail] · ORCID [0000-0000-0000-0000]

[Date]

The Editor-in-Chief
*Journal of King Saud University – Computer and Information Sciences*
Springer Nature

**Submission of an original research article**

Dear Editor-in-Chief,

We submit the manuscript entitled **"Shift-aware probabilities for random convolutional kernel time series classifiers via leverage-moderated leave-one-out calibration"** for consideration as an original research article in the *Journal of King Saud University – Computer and Information Sciences*.

**Summary.** Random convolutional kernel classifiers such as MiniRocket are among the most accurate and efficient time series classifiers, but their ridge readout returns scores rather than probabilities. Existing remedies apply a single global calibration that does not lower confidence when test series drift away from the training data. We propose LOOM, a closed-form readout that standardises the ridge logits by their leverage and fits a temperature and the ridge penalty on exact leave-one-out score–leverage pairs. It needs no held-out data, refitting or ensembles. The same identities give exact jackknife+ prediction sets without refitting. On {{S['n_datasets']}} UCR datasets with five resamples and a pre-registered benchmark of 18 test-time corruptions, {{COVER_RESULT}}

**Fit with the journal.** The work addresses machine learning, pattern recognition and data mining for time series, which are explicitly within the journal's aims and scope. It contributes a reusable, computationally light method and an evaluation protocol of interest to readers working on trustworthy AI in sensor, medical and industrial monitoring applications.

**Originality and relation to prior work.** The closed-form leave-one-out calibration of ridge readouts was introduced by Dempster et al. (PreVal, arXiv:2401.15610). We cite it as the closest prior work and use it as the primary baseline. Our contribution is the per-input leverage moderation fitted on held-out leverages, the closed-form jackknife+ sets, and the evaluation under distribution shift. Negative results (a small in-distribution effect and higher in-distribution ECE caused by target smoothing) are reported.

**Declarations.** This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have approved the submission. The study uses only public benchmark data; no ethics approval was required. Code and all machine-generated results will be made publicly available; an anonymised link is provided for review. The use of an AI assistant is declared in the manuscript, as the journal's policy requires. [Funding and competing interests: to be completed by the authors.]

We confirm that the manuscript has been prepared for double-anonymous review, with author information provided on a separate title page.

Thank you for considering our manuscript.

Yours sincerely,

[Corresponding author name], on behalf of all authors
