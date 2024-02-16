@echo off
for /r %%i in (Main_Seminar_LLM_KnowledgeGraphs_Presentation*.tex) do texify -cp %%i
