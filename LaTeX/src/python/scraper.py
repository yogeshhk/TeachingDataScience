counter = 0
for i in itertools.permutations(string.ascii_lowercase, 8):
    url = "<REDACTED>"+''.join(myhash)+"<REDACTED>"
    r = requests.get(url)
    if (r.status_code != 404):
        print("Found one!", r.status_code, ",", url)
    counter += 1
    if counter % 100 == 0:
        print("Did another 100!", ",", ''.join(myhash))
