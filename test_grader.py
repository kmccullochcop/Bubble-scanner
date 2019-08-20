#built following step by step tutorial as found on
#https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

#install all libs
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


#parser info
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

#answer key to map q to answer
ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}
#number of possible answers per q
ANS = 5

#loading image and converting to grayscale, finding edges
image = cv2.imread(args["image"])

try:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
except cv2.error as error:
    print(error)
    print("could not find image with filename " + args["image"])
    exit()

blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200)

#finding contours in edge map then the contour that corresponds
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts);
docCnt = None

#ensures that we find at least one contour
if len(cnts) > 0:
    #sort contours descending by size
    conts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #loop over sorted contours
    for c in cnts:
        #approx contour len
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        #need 4 points to find paper
        if len(approx) == 4:
            docCnt = approx
            break
        #end 4point IF
    #END c in cnts FOR
#END >0 contour IF

#applying four point perspective, transform image to get top-down view
paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(gray, docCnt.reshape(4,2))

#binarize warped paper
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#find contours in bin img, then inialize list of contours that correspond to Qs
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

#loop over contours
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    #region needs to be large enough to be a q, aspect ratio should be about 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
    #END aspect and region IF
#END contour FOR

#sort q contours top to bottom, then init # of correct answers
try:
    questionCnts = contours.sort_contours(questionCnts, method = "top-to-bottom")[0]
except ValueError as error:
    print(error)
    print("Picture is bad -- cannot detect contours")
    exit()
    
correct = 0

#each q has ANS possible answers, loop over q in batches of ANS
for (q, i) in enumerate(np.arange(0, len(questionCnts), ANS)):
    #sort contours for init q left to right, then init index of bubbled answers
    cnts = contours.sort_contours(questionCnts[i:i+ANS])[0]
    bubbled = None

    flag = False #flag to find if user bubbled more or less than 1 answer

    #loop over sorted contours
    for(j, c) in enumerate(cnts):
        #make mask that reveals only current bubble for the q
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        #apply mask to thresholded img, count # of non-zero px in bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        #checking that user only bubbled one answers
        #most bubbled answers have more than 500 non-zero px

        #end sorted contours loop, have found that more than one answer bubbled
        if flag is True and total > 500:
            bubbled = None
            break
        #END more than one answer IF

        #setting flag to true when bubble found for first time
        if total > 500:
            flag = True
        #END found first bubble IF

        #comparing sum of non-zero px
        if total > 500:
            bubbled = (total, j)
        #END sum IF

        #if curr total > total non-zero px, then we have the bubbled answer
        #if bubbled is None or total > bubbled[0]:
            #bubbled = (total, j)
        #END bubbled answer IF

    #END sorted contours FOR

    #init contour colour and index of correct answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    #check if bubble is correct
    if bubbled and k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
    #END correct answer IF

    if bubbled != None: #only drawing if bubbled has value
        #draw outline of correct answer on test
        cv2.drawContours(paper, [cnts[k]], -1, color, 3)
    #END drawing IF

#END looping over q's FOR

#get test taker

score = (correct/float(ANS))*100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score),(10,30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
