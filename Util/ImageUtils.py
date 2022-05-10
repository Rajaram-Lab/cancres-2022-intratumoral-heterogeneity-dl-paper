"""
This file supports extraction of patches for training. Its functions help in identifying 
annotated regions in the samples.

Copyright (C) 2021, Rajaram Lab - UTSouthwestern 
    
    This file is part of cancres-2022-intratumoral-heterogeneity-dl-paper.
    
    cancres-2022-intratumoral-heterogeneity-dl-paper is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    cancres-2022-intratumoral-heterogeneity-dl-paper is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with cancres-2022-intratumoral-heterogeneity-dl-paper.  If not, see <http://www.gnu.org/licenses/>.
    
    Paul Acosta, Vipul Jarmale 2022
"""
import numpy as np
import xml.etree.ElementTree as ET

def GetXmlAnnoNames(xmlFile):
    tree = ET.parse(xmlFile)
    # get root element
    root = tree.getroot()
    annoNames=[]
    subAnnoNames=[]  
    
    for annoNum,anno in enumerate(root.iter('Annotation')):
        
        annoNames.append(anno.get('Name'))
        regionList = anno.find('Regions')
        regionNames=[]
                 
 

        for item in regionList:
            regionName=item.get('Text')
            if(regionName != None):
                regionNames.append(item.get('Text'))   
            
        subAnnoNames.append(regionNames)        

    return annoNames,subAnnoNames

def ReadRegionsFromXML(xmlFile,layerNum=0):

    # create element tree object
    tree = ET.parse(xmlFile)

    # get root element
    root = tree.getroot()

    
    regionPos = []
    regionNames = []
    isNegative=[]
    regionInfo=[]
    
    
    for annoNum,anno in enumerate(root.iter('Annotation')):

        if(annoNum==layerNum):
            regionList = anno.find('Regions')
        
                 
            for item in regionList:
                # print item.tag,item.attrib
                regionName=item.get('Text')
                if(regionName != None):
                    regionNames.append(item.get('Text'))
                    isNegative.append(item.get('NegativeROA')=='1')
                    idNum=item.get('Id')
                    inputRegionId=item.get('InputRegionId')
                    vertexList = item.find('Vertices')
                    
                    numberOfVertices = sum(1 for i in vertexList)
                    pos = np.zeros((numberOfVertices, 2))
                    counter = 0
                    for v in vertexList:
                        pos[counter, 0] = int(float(v.get('X')))
                        pos[counter, 1] = int(float(v.get('Y')))
                        counter = counter + 1
                    regionPos.append(pos)
                    info={'Length':float(item.get('Length')),'Area':float(item.get('Area')), 
                          'BoundingBox':np.array([np.min(pos[:,0]),np.min(pos[:,1]),np.max(pos[:,0]),np.max(pos[:,1])]),
                          'id':idNum,'inputRegionId':inputRegionId,'Type':float(item.get('Type'))}
                    regionInfo.append(info)
    regionPos = np.array(regionPos)
    isNegative=np.array(isNegative)
    return regionPos,regionNames,regionInfo,isNegative

def GetLoops(pos):
  rowCounter=0
  continueSearch=True
  loops=[]
  while continueSearch:
    idx=np.where(np.logical_and(pos[(rowCounter+1):,0]==pos[rowCounter,0], 
                                    pos[(rowCounter+1):,1]==pos[rowCounter,1]))[0]
    if(len(idx)>0):
      newPos=idx[0]+rowCounter+1
      
      loops.append(pos[rowCounter:min(newPos+1,len(pos)),:])
      rowCounter=newPos+1
      if(rowCounter>=len(pos)):
        continueSearch=False
    else:
      continueSearch=False  
      loops.append(pos[rowCounter:,:])
  return loops      

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def GetQPathTextAnno(annoFile, checkForDisjointAnnos=True):

  with open(annoFile) as f:
      content = f.readlines()
  
  regionPos = []
  regionNames = []
  isNegative=[]
  regionInfo=[]
     
      

  for p in range(len(content)):
    line=content[p]  
    className,data=line.replace(']','').split('[')
    if(isinstance(className,str) and len(className)>0 and (className[0]=='+' or className[0]=='-')):
      className=className  
    else:
      className='+'+className
    regionNames.append(className)
    
    data=np.array([float(x) for x in  data.replace('Point: ','').split(',')])
    coords=np.zeros((int(len(data)/2),2))
    coords[:,0]=data[0::2]
    coords[:,1]=data[1::2]
    if(checkForDisjointAnnos):
      loops=GetLoops(coords)
      loopAreas=[PolyArea(l[:,0],l[:,1]) for l in loops]
      coords=loops[np.argmax(loopAreas)]
    
    regionPos.append(coords)
  
    bbox=np.array([np.min(coords[:,0]),np.min(coords[:,1]),np.max(coords[:,0]),np.max(coords[:,1])])
    l=((bbox[2]-bbox[0])+1)
    w=((bbox[3]-bbox[1])+1)
    area=l*w
    info={'Length':max(l,w),'Area':area,'BoundingBox':bbox,'id':p,'inputRegionId':0,'Type':0}
    regionInfo.append(info)
    
  regionPos=np.array(regionPos)
  isNegative=np.zeros(len(content))==1
  return regionPos,regionNames,regionInfo,isNegative

