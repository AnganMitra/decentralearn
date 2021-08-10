import qarnot

def learnOnCloud():
   conn = qarnot.Connection("sample.conf")
   task = conn.create_task("dfmw-thailand-learn", "docker-network")
   task.constants["DOCKER_REPO"] = "angmit/decentralearn"
   task.constants["DOCKER_TAG"] = "latest"
   task.resources = [ conn.retrieve_bucket("dmfw"), conn.retrieve_bucket("thailandmodels") ]
#    command = "touch /job/end.txt | echo '78'>/job/end.txt | cat /job/end.txt"
   command="bash | ls >> f.txt"
#    command = "ls /job/ " ## place data files for buckets [/job/ auto-sync in Qarnot]
#    command = "ls /job | ls /opt" ## place src files and execution scripts here 
#    command = "cd /opt/ | ./run.sh"
#    command = "python3 /opt/src/main.py -i /job/  -o /job/OnlineModel/  -sdt 2019-03-07 -edt 2019-12-31  -cdt 2019-05-08 -fl 7 -zn 5  -grt cycle  -feat temperature  -resm max -model cnn -plotFig True -modePred True"

   print (command)
   task.constants["DOCKER_CMD"] = command
   task.result = conn.retrieve_bucket("thailandmodels")
   task.submit()

learnOnCloud()
