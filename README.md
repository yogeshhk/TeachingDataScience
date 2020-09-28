# Teaching Data Science

Repository for LaTeX course notes for Python, Machine Learning, Deep Learning, Natural Language Processing, etc. Core content is in the form for Beamer slides, which in turn can get compiled into presentation mode as well as two-column course notes mode.

All tex sources and images have been open sourced as I have taken from others, learnt from others, although I have added some of mine, I need to give it back!! LinkedIn post: https://www.linkedin.com/feed/update/urn:li:activity:6523000857385103360

Copyright (C) 2019 Yogesh H Kulkarni

<!--
## Rational
- Global Recognition and getting opportunities and respect (Part II, 40+, chalk in hand)
- For Giving back from classes to masses, dyan-eshwar. 
- If any $: Donate to CoEP Alumni, Jagriti, Aai for causes (Experfy/INE already)
- For better own understanding
- Tensorflow (Google backing, good production ready tooling)
-->

Avenues: @TFUG Pune, Colleges/Confs, GDE speakers opportunities

## Planned Topics
- RNN [Presentation done]
- Word embeddings [Presentation done]
- MidcurveNN @ DevFests [Presentation done]
- DL in Bioinformatics [Under Construction]
- TBD [References below]

## Playbook (conduct lectures under TFUG Pune)
Logistics:
- Day/time: Saturdays Mornings, 9 to 10 am, else NO
- Pre-Talk:
    - Prep materialas *_seminar.tex and ipynb for 40 minutes talk and then QnA
    - Create Google calendar event with Google Meet link
    - Create Banner png from admin/template.ppt
		- Create TFUG Pune Meetup event with Google Meet link and Banner png
    - Advertise the Meetup link: my LinkedIn, Company Teams, GDE whatsapp grps
    - Copy material pdf/ipynbs under TFUGPune folder, sync with GDrive
- During Talk: Take pictures
- Post-Talk:
    - Fill GDE Advocu entry with picture taken
    - Fill GDE New event survey link
    - Share github sources of the talk on meetup, post pictures there

## Video Recordings
- Using OBS: Open-source Free-ware
	- Sources: Windows Capture
	- Settings:
		- Hot Keys: Alt + 10 to start recording, Alt + F11 to stop
		- Output format : mkv, easier to mix, can be converted to mp4 by File->Remux
		- Output gets stored in Videos folder
- OpenTech to edit video, save to join
- Handbrake to compress video


## LaTex to pdfs

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


<!-- ## Data Science Course Series

<img src="LaTeX/images/teaching_data_science_series.png"/> -->

### How to Run LaTeX:
* Driver files for the courses are named with "Main_Workshop/Seminar_<course>_CheatSheet/Presentation.tex"
* Both the Cheatsheet (meaning course notes in two column format) and Presentation.tex refer to same core content file, which in turn contains are the topic files.
* Run make bat for the course you need. Inside, its just a texify command, so you can modify it as per your OS.
* You can compile individual "Main_Workshop/Seminar_<course>_CheatSheet/Presentation.tex" also using your LaTeX system.
* Instead of these given driver files, you can have your own main files and include just the *content.tex files.

## References to be used for updating the Training Material
* Representation Learning - Yoshua Bengio https://www.youtube.com/watch?v=O6itYc2nnnM
* Jordan Boyd-Graber ML NLP Play list https://www.youtube.com/user/ezubaric/playlists
* Shala 2020 IITB https://www.youtube.com/channel/UCobe_Yc7nV6kux94A5doIiA/videos
*	Visual Notes, Graphics Recording
	* Graham Shaw https://www.youtube.com/user/VisionLearning/videos
	* 50+ Awesome Resources to Create Visual Notes, Graphic Recordings & Sketchnotes https://creativemarket.com/blog/50-awesome-resources-to-create-visual-notes-graphic-recordings-sketchnotes
	* Bullet Journal / Planner Icon Doodles | Doodle with Me https://www.youtube.com/watch?v=o1zdgGGUtNo&list=PLx5c844mTRMAKUS2eJJWpKgSJnUgKrLpS
	* Learning Graphic Facilitation - 7 Elements by Bigger Picture https://www.youtube.com/watch?v=S5DJC6LaOCI
*	Mathematics
    * Essence of linear algebra - 3Blue1Brown https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
    * Statistics Brandon Foltz https://www.youtube.com/user/BCFoltz/playlists
    * Intro - The Math of Intelligence – YouTube
    * StatQuest: https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw
    * Mathematical Monk https://www.youtube.com/channel/UCcAtD_VYwcYwVbTdvArsm7w
    * Seeing Theory https://seeing-theory.brown.edu/index.html 
    * Rahul Patwari https://www.youtube.com/user/oldblueday/playlists 
*	Python
    * Real Python Tutorials https://realpython.com/ nice cartoon tutorials on various subtopics
	* Advanced Python or Understanding Python – YouTube https://www.youtube.com/watch?v=E_kZDvwofHY 
    * Introduction - Intermediate Python Programming p. 1 – YouTube https://www.youtube.com/watch?v=YSe9Tu_iNQQ&list=PLQVvvaa0QuDfju7ADVp5W1GF9jVhjbX-_ 
    * Python Programming Tutorials https://pythonprogramming.net/  
    * Google's Python Class | Python Education | Google Developers https://www.youtube.com/playlist?list=PL5-da3qGB5IA5NwDxcEJ5dvt8F9OQP7q5   
    * Python https://www.youtube.com/watch?v=YYXdXT2l-Gg&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU 
*	Machine Learning
    * AWS QwikLabs
        * Applied Machine Learning: Building Models for an Amazon Use Case | Qwiklabs
        * Build a Dynamic Conversational Bot - Part 1 | Qwiklabs
    * Georgia Tech - Machine Learning Udacity https://www.youtube.com/watch?v=Ki2iHgKxRBo&list=PLAwxTw4SYaPl0N6-e1GvyLp5-MUMUjOKo 
    * ML Victor Levrenko https://www.youtube.com/user/victorlavrenko/playlists
    * Thales Sehn Körting ML https://www.youtube.com/channel/UCSd_7rz5nzSnzUYbjaCXC5g
    * How Machine Learning works https://www.youtube.com/user/tkorting/playlists 
    * Statistics ML https://www.youtube.com/user/BCFoltz/playlists 
    * dive-into-machine-learning http://hangtwenty.github.io/dive-into-machine-learning/ 
    * General Assembly's Data Science course JustMarkham https://github.com/justmarkham/DAT8#class-1-introduction-to-data-science 
    * Data School In-depth introduction to machine learning in 15 hours of expert videos https://www.dataschool.io/15-hours-of-expert-machine-learning-videos/ 
    * Machine Learning Mastery by Jason Brownlee https://machinelearningmastery.com/ 
    * Machine Learning guides: http://ml4a.github.io/guides/ 
    * 20 Best YouTube channels for AI and machine Learning https://gengo.ai/articles/20-best-youtube-channels-for-ai-and-machine-learning/ 
    * Code Repository Py+ML by Sebastian Raschka https://github.com/rasbt/python-machine-learning-book-2nd-edition 
*	Deep Learning
    * Deep learning Book Ian GoodFellow https://www.youtube.com/watch?v=vi7lACKOUao&list=PLsXu9MHQGs8df5A4PzQGw-kfviylC-R9b
    * Deep Learning by Google https://in.udacity.com/course/deep-learning--ud730
    * Deep Learning Book lectures https://www.youtube.com/channel/UCF9O8Vj-FEbRDA5DcDGz-Pg/playlists
    * Neural Networks https://www.youtube.com/playlist?list=PL29C61214F2146796
    * Neural Networks Demystified https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU 
    * How Deep Neural Network works https://www.youtube.com/watch?v=WCUNPb-5EYI&list=PLVZqlMpoM6kbaeySxhdtgQPFEC5nV7Faa 
    * What is Deep Learning Edureka https://www.youtube.com/watch?v=dafuAz_CV7Q&list=PL9ooVrP1hQOEX8BKDplfG86ky8s7Oxbzg
    * Deep Learning Decal Fall 2017 Berkeley https://www.youtube.com/playlist?list=PLzWRmD0Vi2KXcrTVBSK2w-VyjAAfNaqgF
*	NLP
    * Information Retrieval  https://www.youtube.com/watch?v=nfoudtpBV68&list=PLiNErZ5Bus8qNxNsFZFkh-9_CzZRW9iH9 
    * Natural Language Processing with Deep Learning (http://web.stanford.edu/class/cs224n/ )
    * https://www.linkedin.com/pulse/introduction-natural-language-processing-nlp-2016-kapil-khangaonkar 
    * NLP and all : https://www.cs.bgu.ac.il/~elhadad/nlp18.html
    * Standord NLP https://www.youtube.com/watch?v=nfoudtpBV68&list=PL6397E4B26D00A269
    * From Languages to Information https://www.youtube.com/channel/UC_48v322owNVtORXuMeRmpA/playlists?view=50&sort=dd&shelf_id=2 https://web.stanford.edu/class/cs124/ 
		* Analytics Vidhya NLP Learning Path 2020 https://www.analyticsvidhya.com/blog/2020/01/learning-path-nlp-2020/?utm_source=feedburner
    * LinkedIn refs
        * 1. Machine Translation (https://lnkd.in/fAYvEne)
        * 2. Question Answering (Like Chat-bot) (https://lnkd.in/fFZmP4f)
        * 3. Sentiment Analysis (https://lnkd.in/fUDGAQW)
        * 4. Text Search (with Synonyms) (https://lnkd.in/fnU_a_H)
        * 5. Text Classifications (https://lnkd.in/f8mjKAP)
        * 6. Spelling Corrector (https://lnkd.in/f8JXNUv)
        * 7. Entity (Person, Place, or Brand) Recognition (https://lnkd.in/f2fzgAa)
        * 8. Text Summarization (https://lnkd.in/fdzWqXC)
        * 9. Text Similarity (https://lnkd.in/fv_sWuM)
        * 10. Topic Detection (https://lnkd.in/fxmhJZc)
        * 11. Emotion Recognition (https://lnkd.in/fK4m66Q)
        * 12. Language Identification (https://lnkd.in/fqfjxF9)
        * 13. Document Ranking (https://lnkd.in/fJZnkqz)
        * 14. Fake News Detection (https://lnkd.in/fkrkF8Q)
*	Deep NLP
    * CS224n: Natural Language Processing with Deep Learning – Stanford https://www.youtube.com/playlist?list=PLqdrfNEc5QnuV9RwUAhoJcoQvu4Q46Lja 
    * Graham Neubig Deep NLP https://www.youtube.com/user/neubig/playlists
*	AI
    * Artificial Intelligence, NPTEL https://www.youtube.com/watch?v=XCPZBD9lbVo&list=PLbMVogVj5nJQu5qwm-HmJgjmeGhsErvXD 
    * AI UC Berkeley https://www.youtube.com/watch?v=W1S-HSakPTM&list=PL6MuV0DF6AuoviA41dtji6q-PM4hvAcNk
    * Nick Hawes AI https://vimeo.com/125049035 
*	Chatbot
    * Building a chatbot with Rasa NLU and Rasa Core https://vimeo.com/254777331
    * Getting started with Rasa: using the Rasa Stack starter-pack https://www.youtube.com/watch?v=lQZ_x0LRUbI
    * Say hi to Sara - our Rasa Demo Bot! (alpha) https://rasa.com/docs/demobot/
    * Conversational AI with Rasa Core & NLU - Tom Bocklisch https://www.youtube.com/watch?v=zRqjH7fT0G0 
    * Deprecating the state machine: building conversational AI with the Rasa stack - Justina Petraitytė https://www.youtube.com/watch?v=3qgWQ-u1lQo
    * Introduction to Conversational Software https://campus.datacamp.com/courses/building-chatbots-in-python/chatbots-101?ex=1 
*	All
    * Data School : https://www.youtube.com/channel/UCnVzApLJE2ljPZSeQylSEyg 
    * Data Science Decal Fall 2017 Berkeley https://www.youtube.com/playlist?list=PLzWRmD0Vi2KUE2Nopo2JD4NE2_FDrxPQr
    * Python ML DL NLP - Sentdex https://www.youtube.com/user/sentdex/playlists 
    * Free Data Science Resources Cloud ML http://www.claoudml.co/
    * Microsoft Professional Program for Artificial Intelligence track https://academy.microsoft.com/en-us/tracks/artificial-intelligence/
    * ML DL Maths Luis Serrano https://www.youtube.com/channel/UCgBncpylJ1kiVaPyP-PZauQ/playlists
    * Kaggle Competitions NYC https://www.youtube.com/playlist?list=PL0NklAB2PfRRO1QPx78IYLgfvRpw4OILl
    * Madhu Sanjeevi https://medium.com/@madhusanjeevi.ai
    * MIT AGI: Artificial General Intelligence https://www.youtube.com/watch?v=-GV_A9Js2nM&list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4
    * Open Data Science Masters http://datasciencemasters.org/
    * Over 150 of the Best Machine Learning, NLP, and Python Tutorials I’ve Found 
    * Data Camp: https://www.youtube.com/user/dataschool
    * GeekForGeeks https://www.youtube.com/watch?v=v4cd1O4zkGw
    * MIT 6.034 Artificial Intelligence, Fall 2010 https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi 
    * Semicolon Python ML DL https://www.youtube.com/channel/UCwB7HrnRlOfasrbCJoiZ9Lg/playlists
    * Siraj Maths ML DL https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A/playlists
    * Victor Lavrenko Text mining, ML, DL https://www.youtube.com/user/victorlavrenko/playlists
    * Well Academy Python AI https://www.youtube.com/channel/UCQA9tK0nRK1e_Bqg0uETs8A/playlists


## Disclaimer:
* Author (yogeshkulkarni@yahoo.com) gives no guarantee of the correctness of the content. Notes have been built using lots of publicly available material. 
* Although care has been taken to cite the original sources as much as possible, but there could be some missing ones. Do point them and I will update wherever possible. 
* Lots of improvements are still to be made. So, don’t depend on it at all, fully. 
* Do send in your suggestions/comments/corrections/Pull-requests.