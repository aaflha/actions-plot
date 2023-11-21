# actions-plot :kissing_heart:

```
værsågod krifla
```

The image to embed has the following url: https://raw.githubusercontent.com/aaflha/actions-plot/main/fig.png

### Get going
You will need `python` 3.x, `venv` and `git` to get going
|||
|--|--|
|clone repo:|`git clone https://github.com/aaflha/actions-plot.git`|
|create virtual env|`python3 -m venv venv`|
|activate virtual env|`. venv/bin/activate`|
|install dependencies|`pip install -r requirements.txt`|

### Contributing
|||
|--|--|
|make your change(s)|...|
|if new dependencies added|`pip freeze > requirements.txt`|
|ensure local is up to date with remote|`git pull`|
|add change(s) to staging|`git add <your file(s)>`|
|commit changes|`git commit -m "<your descriptive commit message>"`|
|push to remote|`git push`|

### Changing the schedue
The action is scheduled using cron syntax in .github/workflows/actions.yml line 5. Consult https://crontab.guru/ for guidance.