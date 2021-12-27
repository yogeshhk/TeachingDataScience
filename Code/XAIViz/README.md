# Visualization for Explain-ability

<!-- As AI goes more and more AutoML it becomes black box more and more. Deep Learning is inherently non transparent. This hurts debugging and impediments justifying predictions. It has become legally imperative to back-trace the predictions.

Explainable AI (XAI) has approaches to deal with this problem. Although there are mathematical techniques like SHAPly values and LIME for XAI, stress here would be to achieve explain-ability via Visualization and insights that come through pictures and animations. -->

Concepts or algorithms are hard to understand, many a times. Typically due unavailability of good explanation. Just reading may not ingest the understanding. Visualization helps. Using animation is better.

Code-bases in this folder attempt to show such examples.


- What it is not: UI, UX or HCI design
- What it is: explain-ability via visualization
- Application to: AI-ML, NLP, Research papers, Education, Dashboards, etc.

Copyright (C) 2021 Yogesh H Kulkarni

## Principles

Structuring Math explanations - by Grant Sanderson, https://www.youtube.com/watch?v=ojjzXyQCzso
- Concrete before Abstract
  - Give examples first, then theory
	- Be aware of abstractions:
	
		![Layers](./Manim/images/layers.png)
		
		![Layers Example](./Manim/images/layersexample.png)

- Topic choice > production quality.
	- Content is king, how you produce it is secondary.
	- Choose wisely, needs to be fresh
	- But no 0 production quality though. Good sound a must.
- Be niche
	- Choose something rare, esoteric
	- Cant compete with other million videos.
	- Build small, loyal audience
	- May appeal to more than what you think.
- Know your genre
	- Be upfront about your level, discovery mode
	- Helping others, tutoring mode
	- Slow/Fast? Don't follow others as is
- Definitions not at the beginning

	
## References
- Papers:
  - https://visxai.io/
  - https://distill.pub/
- Groups:
  - https://research.google/teams/brain/pair/
  - [Explainable AI + VIS](http://vis.cse.ust.hk/groups/xai-vis/)
- Conferences:
	- [Visualization for AI Explainability](https://visxai.io/)
- Libraries
	- Explainer Dashboard [Link](https://www.linkedin.com/posts/greg-coquillo_datascience-machinelearning-artificialintelligence-activity-6878763723788566528-dqsE)

		Quickly build Explainable AI Dashboards that show the inner workings of so-called "blackbox" machine learning models!

		Do GDPR policies apply to you? Then this package might be useful. Remember, not all models need to be explained. For example, as a user I don’t have to know why models that power autonomous vehicles work the way they work. I just need to know the vehicles are safe to use.

		But when Explainable AI is needed, please note it goes beyond explaining importance of features. X-AI empowers non-data scientists such as managers and executives to be able to inspect and explain model behaviors while removing dependency on data scientists.

		The Explainer Dashboard provides:
		- Interactive plots of model performance
		- Feature contributions to individual predictions
		- Feature importance
		- Interactive SHAP values
		- Goodness of fit plots for regression models
		- ROC AUC plots, Confusion Matrix for classifiers
		- Visualization of individual trees for Random Forest and XGBoost
		- ”What-If” analysis
		- Partial dependence plots

		You can do so much more with this package such as exploring directly from your notebook, building custom dashboards and combining into a single ExplainerHub.

		Dashboards can be exported to static html directly from a running dashboard, or programmatically as an artifact as part of an automated CI/CD deployment process.

		Hope you find this very useful and increase X-AI launch speed for your use cases!

		Thank you Oege Dijk for this awesome package.

