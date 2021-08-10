import qarnot

# def learnOnCloud():
#    conn = qarnot.Connection("sample.conf")
#    task = conn.create_task("dfmw-thailand-learn", "docker-batch")
#    task.constants["DOCKER_REPO"] = "angmit/decentralearn"
#    task.constants["DOCKER_TAG"] = "latest"
#    task.resources = [ conn.retrieve_bucket("dmfw") ]
# #    command = "ls /job/ " ## place data files for buckets [/job/ auto-sync in Qarnot]
# #    command = "ls /job | ls /opt" ## place src files and execution scripts here 
# #    command = "cd /opt/ | ./run.sh"
#    command = "python3 /opt/src/main.py -i /job/  -o /job/OnlineModel/  -sdt 2019-03-07 -edt 2019-12-31  -cdt 2019-05-08 -fl 7 -zn 5  -grt cycle  -feat temperature  -resm max -model cnn -plotFig True -modePred True"

#    print (command)
#    task.constants["DOCKER_CMD"] = command

#    task.result = conn.retrieve_bucket("dmfw")
#    task.submit()

# learnOnCloud()

def runExperiment(floorNumber, graph, feature, ):
   conn = qarnot.Connection("sample.conf")
   task = conn.create_task(f"dfmw-{floorNumber}-{graph}-{feature}", "docker-network")
   task.constants["DOCKER_REPO"] = "angmit/decentralearn"
   task.constants["DOCKER_TAG"] = "latest"
   task.resources = [ conn.retrieve_bucket("dmfw") ]
#    command = "ls /job/ " ## place data files for buckets [/job/ auto-sync in Qarnot]
#    command = "ls /job | ls /opt" ## place src files and execution scripts here 
#    command = "cd /opt/ | ./run.sh"
   command = f"python3 /opt/src/main.py -i /job/  -o /job/{floorNumber}-{graph}-{feature}-OnlineModel/  -sdt 2019-03-07 -edt 2019-12-31  -cdt 2019-05-08 -fl {floorNumber} -zn 5  -grt {graph}  -feat {feature}  -resm max -model cnn -plotFig True -modePred True"

   print (command)
   task.constants["DOCKER_CMD"] = command

   task.result = conn.retrieve_bucket("dmfw")
   task.submit()



if __name__=="__main__":
    # learnOnCloud() # test one task
    for feature in ["temperature", "humidity", 'ACPower','lightPower','appPower','lux']:
        for graph in [ "cycle", "complete","grid", "line"]:
            for floorNumber in [3,4,5,7]:
                runExperiment(floorNumber, graph, feature, )
            

