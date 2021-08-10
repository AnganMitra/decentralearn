import qarnot
import os

def pushToQBucket(localDirectory, qDirectory, skipDirectoryInclude = False):
   conn = qarnot.Connection("sample.conf")
   bucket = conn.retrieve_bucket("thailandmodels")
   files = sorted(os.listdir(localDirectory), key=lambda fn:os.path.getctime(os.path.join(localDirectory, fn)))
   print (files)
   for file in files:
       print (file)
       if os.path.isdir(localDirectory+file):
           if skipDirectoryInclude : continue
           bucket.add_directory(localDirectory+file, qDirectory+file)
           for level2file in os.listdir(localDirectory+file):
               try:
                   bucket.add_file(localDirectory+file+"/"+level2file, qDirectory+file+"/"+level2file )
               except: 
                   pass
       else:
           bucket.add_file(localDirectory+file, qDirectory+file)

if __name__=="__main__":
    pushToQBucket("./src/", "Qtest/")
