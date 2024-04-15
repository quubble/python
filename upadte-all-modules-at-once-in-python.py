# update-all-python-packages-at-once USING PIP Powershell.py

C:;cd C:\Users\asus\AppData\Local\Programs\Python\Python312;python -m pip install --upgrade pip;pip list --outdated;pip freeze | %{$_.split('==')[0]} | %{pip install --upgrade $_}

#pip freeze > requirements.txt
#pip freeze > requirements.txt
