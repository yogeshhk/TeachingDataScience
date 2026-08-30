$sessions = @(
    @{num=1; short="AI_Overview"}
    @{num=2; short="Python_Overview"}
    @{num=3; short="EDA_DataPrep"}
    @{num=4; short="Pandas"}
    @{num=5; short="ML_Intro"}
    @{num=6; short="ML_Concepts"}
    @{num=7; short="Sklearn_Workflow"}
    @{num=8; short="Linear_Regression"}
    @{num=9; short="Logistic_Regression"}
    @{num=10; short="Decision_Trees"}
    @{num=11; short="Ensemble_RF"}
    @{num=12; short="SVM"}
    @{num=13; short="Naive_Bayes"}
    @{num=14; short="KNN"}
    @{num=15; short="KMeans"}
    @{num=16; short="PCA"}
    @{num=17; short="Titanic_Capstone"}
    @{num=18; short="MLOps_Deployment"}
    @{num=19; short="ME_Apps"}
)

$presTemplate = '\documentclass[xcolor=dvipsnames,compress,t,pdf,9pt]{beamer}
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

\end{document}'

$cheatTemplate = '\documentclass[8pt,landscape]{article}
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

\end{document}'

foreach ($s in $sessions) {
    $presContent = $presTemplate -replace 'SESSION', $s.num
    $presFile = "Main_Course_ML_CoEP_Session_$($s.num)_$($s.short)_Presentation.tex"
    $presContent | Out-File -FilePath $presFile -Encoding UTF8
    
    $cheatContent = $cheatTemplate -replace 'SESSION_NUM', $s.num -replace 'SESSION_NAME', $s.short -replace 'SESSION', $s.num
    $cheatFile = "Main_Course_ML_CoEP_Session_$($s.num)_$($s.short)_CheatSheet.tex"
    $cheatContent | Out-File -FilePath $cheatFile -Encoding UTF8
    
    Write-Host "Created $presFile and $cheatFile"
}

Write-Host "Done: 38 driver files created (19 Presentation + 19 CheatSheet)"
