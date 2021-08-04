from lib import *



if __name__ == '__main__':

    parser = argp.ArgumentParser()

    parser.add_argument("-i", "--input_data", help="path of the data folder")
    parser.add_argument("-o", "--output_dir", help="path of the output directory.")
    parser.add_argument("-sdt", "--start_date", help="True to reload a previous computation, present in the output path else not needed.")
    parser.add_argument("-edt", "--end_date", help="True to generate a topological graph")
    parser.add_argument("-cdt", "--cut_date", help="True to compute a hull")
    parser.add_argument("-fl", "--floor", help="True to generate floorplans storeywise")
    parser.add_argument("-zn", "--nb_zone", help="Buffer margin as a float for hull computation")
    

    args = vars(parser.parse_args())
    ifc_file = None
    output_dir = None
    try:
        ifc_file = args['input_file']
        print ("IFC File : ", ifc_file)
    except:
        pass
    try:
        output_dir = args['output_dir']
        print ("Output Directory : ",output_dir)
    except:
        pass
    
    if args['reload']:
        GlobalElements.reload_mode = True
    try:
        # import pdb; pdb.set_trace()
        os.mkdir(output_dir)
        os.mkdir(output_dir+'floorplan/')
        os.mkdir(output_dir+'hull/')
        os.mkdir(output_dir+'graph/')
        os.mkdir(output_dir + 'json/')
    except:
        print("Directory ",output_dir, " exists. Over writing ...")
        # input("Press Enter to continue or press Ctrl + D to exit the program...")

    
    if not GlobalElements.reload_mode:
        ifc_file = ifcopenshell.open(ifc_file)
        print ('Lexical Parsing ....')
        parseLexical(ifc_file)
        print ('Geometric Parsing ....')
        parse3D(ifc_file)
        createStoreyOrder()
    else:
        print ('Reading Product Dictionary and configuration files ....')
        reloadComputation(output_dir+ 'json/')

    if args['floorplan_gen']:
        print ('Render floorplan ....')
        floorplanSVG(output_dir + 'floorplan/', all = False )          # Put this to true to generate svg for all the cuttings
    
    if args['hull_gen']:
        print ('Render Concave Hull ....')
        EPS = 1e-1
        if args['EPS']: EPS = float(args['EPS'])
        # print (EPS)
        ConcaveHull(output_dir + 'hull/',EPS,with_spaces= True)
    dumpBuildingJson(output_dir + 'json/')
    
    if not GlobalElements.reload_mode:
        print ('Config Dump ....')        
        dumpConfig(output_dir + 'json/')
    if args['graph_gen']:
        print ('Topology Graph Generation ....')
        GenerateTopologicalGraph(output_dir ,'graph/' ,'json/')
        
    print ('Q Json Dump ....')
    dumpStoreyJson(output_dir+ 'json/')

    
