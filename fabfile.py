from fabric.api import local

miniprez = "python ~/git-repo/miniprez/miniprez"

def run():
    local("{} index.md".format(miniprez))

def edit():
    local("emacs index.md &")

def watch():
    local("{} --watch=1 index.md".format(miniprez))

def view():
    local("xdg-open index.html")

def push():
    local("git commit -a")
    local("git push")
