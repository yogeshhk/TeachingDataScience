#!/bin/bash

sessions=(
"1:AI_Overview"
"2:Python_Overview"
"3:EDA_DataPrep"
"4:Pandas"
"5:ML_Intro"
"6:ML_Concepts"
"7:Sklearn_Workflow"
"8:Linear_Regression"
"9:Logistic_Regression"
"10:Decision_Trees"
"11:Ensemble_RF"
"12:SVM"
"13:Naive_Bayes"
"14:KNN"
"15:KMeans"
"16:PCA"
"17:Titanic_Capstone"
"18:MLOps_Deployment"
"19:ME_Apps"
)

pres_template=$(cat <<'PRES_EOF'
\documentclass[xcolor=dvipsnames,compress,t,pdf,9pt]{beamer}
\input{template_presentation}
\setbeameroption{hide notes}
\graphicspath{{./images/}}
\title[\insertframenumber /\inserttotalframenumber]{Course ML CoEP}

\begin{document}

	\begin{frame}
	\titlepage
	\end{frame}

	\begin{frame}{Outline}
	    \tableofcontents
	\end{frame}

	\input{about_me}
	\input{about_me_quantum}
	\input{course_mlcoep_session_SESSION_content}
	\input{thanks}

\end{document}
PRES_EOF
)

cheat_template=$(cat <<'CHEAT_EOF'
\documentclass[8pt,landscape]{article}
\input{template_cheatsheet}
\graphicspath{{images/}}

\begin{document}
\footnotesize

\begin{center}
\Large{\textbf{Machine Learning for Mechanical Engineers at CoEP\ Session SESSION_NUM: SESSION_NAME}}
\end{center}

\begin{multicols}{2}
\input{course_mlcoep_session_SESSION_content}
\end{multicols}

\rule{\linewidth}{0.25pt}
\scriptsize
Copyleft \textcopyleft\  Send suggestions to
\href{http://www.yogeshkulkarni.com}{yogeshkulkarni@yahoo.com}

\end{document}
CHEAT_EOF
)

for session in "${sessions[@]}"; do
    IFS=':' read -r num short <<< "$session"
    
    # Presentation driver
    pres_content="${pres_template//SESSION/${num}}"
    pres_file="Main_Course_ML_CoEP_Session_${num}_${short}_Presentation.tex"
    echo "$pres_content" > "$pres_file"
    echo "Created $pres_file"
    
    # CheatSheet driver
    cheat_content="${cheat_template//SESSION_NUM/${num}}"
    cheat_content="${cheat_content//SESSION_NAME/${short}}"
    cheat_content="${cheat_content//SESSION/${num}}"
    cheat_file="Main_Course_ML_CoEP_Session_${num}_${short}_CheatSheet.tex"
    echo "$cheat_content" > "$cheat_file"
    echo "Created $cheat_file"
done

echo "Done: 38 driver files created"
