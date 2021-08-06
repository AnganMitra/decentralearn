import qarnot

conn = qarnot.Connection("qarnot.conf")


def learnOnCloud():
   conn = qarnot.Connection("sample.conf")
   task = conn.create_task("dfmw-thailand-learn", "docker-batch")
   task.constants["DOCKER_REPO"] = "angmit/decentralearn"
   task.constants["DOCKER_TAG"] = "latest"
   task.resources = [ conn.retrieve_bucket("dfmw") ]
   command = "cd /opt/ |./run.sh" 
   # command = "ls"
   print (command)
   task.constants["DOCKER_CMD"] = command

   task.result = conn.retrieve_bucket("dfmw")
   task.submit()

learnOnCloud()