# Teaching Data Science

Repository for LaTeX course notes for Python, Machine Learning, Deep Learning, Natural Language Processing, etc. Core content is in the form for Beamer slides, which in turn can get compiled into presentation pdf as well as two-column course notes pdf.

All tex sources and images have been open sourced as I have built them by referring to material from others, learnt from others, although I have added some of mine, I need to give it back!!

LinkedIn post: https://www.linkedin.com/feed/update/urn:li:activity:6523000857385103360

- **Mission**: To spread knowledge of Data Science related subjects to a wider audience.
- **Vision**: Let many participate in the industry which is aiming for "From Automation, To Autonomy!!"
- **Values**: Giving back, paying it forward!!
- **Goal**: "From Automation to Autonomy"!!

Intent is:
"आपणासी जे जे ठावे| ते ते दुसऱ्यासी सांगावे| 
शहाणे करून सोडावे| सकळ जन. 
- समर्थ रामदास स्वामीं, 

but care has to be taken:
"अभ्यासे प्रगट व्हावे। 
नाहीतरी झाकोनि असावे। 
प्रगट होऊनि नासावे। 
हे बरे नव्हे।। 
- समर्थ रामदास स्वामीं"

Copyright (C) 2019 Yogesh H Kulkarni

## Steps for LaTeX source files to pdfs

### Code Arrangement
*	LaTeX directory 
	* Has tex sources along with necessary images
	*	Naming: subject_maintopic_subtopic.tex eg maths_linearalgebra_matrices.tex
	*	Main_Workshop/Seminar_Presentation/CourseMaterial.tex are the driver files
	*	They intern contain common content files, which have included actual source files
	*	Make bat files compile the appropriate sources
*	Code directory 
	*	Has running python or ipython notebook files with necessary images and data
	*	Naming should be such that corresponding latex file can be associated
	*	Library based tex file, say, sklearn_decisiontree.tex will have just template code and short fully working examples from Mastering Machine Learning whereas the sklearn_decisiontree.ipynb will have running workflows. No need to sync both. You may want to keep ipynb’s pdf in LaTeX/images directory for reference
*	References directory (not uploaded, as it is mostly from others github repos, nothing much of mine)
	*	Has papers, code, ppts as base material to be used for content preparation

### Requirements
* LaTeX (tested with MikTex 2.9 on Windows 7, 64bit)
* Need to install LaTeX packages, as and when you get such warning/suggestions.
* Using TexWorks as IDE

### How to Run LaTeX:
* Driver files for the courses are named with "Main_Workshop/Seminar_<course>_CheatSheet/Presentation.tex"
* Both the Cheatsheet (meaning course notes in two column format) and Presentation.tex refer to same core content file, which in turn contains are the topic files.
* Run make bat for the course you need. Inside, its just a texify command, so you can modify it as per your OS.
* You can compile individual "Main_Workshop/Seminar_<course>_CheatSheet/Presentation.tex" also using your LaTeX system.
* Instead of these given driver files, you can have your own main files and include just the *content.tex files.

<!-- ## Notes

<!-- ## Good resources for learning
*	Machine Learning
    * ML Victor Levrenko https://www.youtube.com/user/victorlavrenko/playlists
    * Statistics ML https://www.youtube.com/user/BCFoltz/playlists 
*	Deep Learning
    * Deep Learning by Google https://in.udacity.com/course/deep-learning--ud730
    * Deep Learning Book lectures https://www.youtube.com/channel/UCF9O8Vj-FEbRDA5DcDGz-Pg/playlists

*	General
    * Open Data Science Masters http://datasciencemasters.org/
    * GeekForGeeks https://www.youtube.com/watch?v=v4cd1O4zkGw
 -->

## Disclaimer:
* Author (firstnamelastname at yahoo dot com) gives no guarantee of the correctness of the content. Notes have been built using lots of publicly available material. 
* Although care has been taken to cite the original sources as much as possible, but there could be some missing ones. Do point them and I will update wherever possible. 
* Lots of improvements are still to be made. So, don’t depend on it at all, fully. 
* Do send in your suggestions/comments/corrections/Pull-requests.
