# Teaching Data Science
LaTeX course notes for Python, Machine Learning, Deep Learning, Natural Language Processing, etc. Core content is in the form for Beamer slides, which in turn can get compiled into presentation mode as well as two-column course notes mode.

Copyright (C) 2019 Yogesh H Kulkarni

## Requirements:
* LaTeX (tested with MikTex 2.9 on Windows 7, 64bit)
* Need to install LaTeX packages, as and when you get such warning/suggestions.

## How to Run:
* Driver files for the courses are named with "Main_Workshop/Seminar_<course>_CheatSheet/Presentation.tex"
* Both the Cheatsheet (meaning course notes in two column format) and Presentation.tex refer to same core content file, which in turn contains are the topic files.
* Run make bat for the course you need. Inside, its just a texify command, so you can modify it as per your OS.
* You can compile individual "Main_Workshop/Seminar_<course>_CheatSheet/Presentation.tex" also using your LaTeX system.
* Instead of these given driver files, you can have your own main files and include just the *content.tex files.

## Disclaimer:
* Author (yogeshkulkarni@yahoo.com) gives no guarantee of the correctness of the content. Notes have been built using lots of publicly available material. 
* Although care has been taken to cite the original sources as much as possible, but there could be some missing ones. Do point them and I will update wherever possible. 
* Lots of improvements are still to be made. So, don’t depend on it at all, fully. 
* Do send in your suggestions/comments/corrections/Pull-requests.