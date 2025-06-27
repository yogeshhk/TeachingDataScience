@echo off
for /r %%i in (Main_Seminar_NLP_WordEmbeddings_*.tex) do texify -cp %%i
