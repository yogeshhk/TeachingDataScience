#!/usr/bin/python

# The package `gitlab` is not listed as dependency in our Pipfile and should be installed globally.
import gitlab

# This uses your local GitLab configuration, check the documentation of GitLab on how to set this up.
gl = gitlab.Gitlab.from_config('gitlab-ewi')
# The project ID can be found on the project homepage
project = gl.projects.get(9)
issues = project.issues

milestonedict = dict()
labeldict = dict()

for ms in project.milestones.list():
    milestonedict[ms.title] = ms.id

titles = [
]

for title in titles:
    newIssue = dict()
    newIssue['title'] = title
    newIssue['milestone_id'] = milestonedict["Assignment 7"]
    newIssue['labels'] = ["assignment", "implementation"]
    newIssue['assignee_ids'] = [38]  # Your own user-id can be found at https://gitlab.ewi.tudelft.nl/profile
    issues.create(newIssue)

for issue in issues.list(scope='all', all=True, state='opened'):
    foundHug = False
    for p in issue.assignees:
        # Change this line if you want to look for issues assigned to someone other than 'mrhug' (id:4)
        if p['id'] == 4:
            foundHug = True
    if foundHug:
        print("Subscribing to", issue.id)
        try:
            issue.subscribe()
        except gitlab.exceptions.GitlabSubscribeError:
            pass
    else:
        print("Unsubscribing to", issue.id)
        try:
            issue.unsubscribe()
        except gitlab.exceptions.GitlabUnsubscribeError:
            pass
