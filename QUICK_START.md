# Quick Start Guide - Sign Language Translation

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
cd /home/runner/work/SignL/SignL
pip install -r requirements.txt
```

**Key dependencies:**
- mediapipe (hand tracking)
- opencv-python (video processing)
- fastapi (web server)
- numpy (calculations)

### Step 2: Start the Server
```bash
./start.sh
```

Or manually:
```bash
python -m uvicorn signl.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Open Browser
```
http://localhost:8000/static/index.html
```

Grant camera permission when prompted.

---

## 🎯 Using Sign Language Translation

### Real-Time Translation

1. **Position yourself**
   - Face the camera
   - Ensure good lighting
   - Keep hand in frame

2. **Make a sign**
   - Clear, distinct gestures
   - Hold for 1-2 seconds
   - One sign at a time

3. **Watch translation**
   - Appears in right panel
   - Timestamped
   - Confidence score shown
   - Audio plays automatically

### Example Signs to Try

#### Easy Signs
1. **Thumbs Up** 👍
   - Close all fingers
   - Extend thumb upward
   - Recognition: "thumbs_up"

2. **Peace Sign** ✌️
   - Extend index and middle fingers
   - Keep other fingers closed
   - Recognition: "peace"

3. **Okay Sign** 👌
   - Touch thumb and index finger
   - Extend other fingers
   - Recognition: "okay"

4. **Stop Sign** ✋
   - Open palm
   - All fingers extended
   - Face palm to camera
   - Recognition: "stop"

#### ASL Letters
1. **Letter A**
   - Make a fist
   - Thumb wrapped over fingers
   - Recognition: "A"

### Testing with Video

1. **Click "Test Video" button**
   - Located in top control bar
   - Blue button with 📹 icon

2. **Select video file**
   - Formats: MP4, AVI, MOV
   - Max size: 100MB recommended
   - Best: Clear sign language video

3. **Wait for processing**
   - Upload: ~5-10 seconds
   - Process: ~1-2 seconds per second of video
   - Results appear automatically

4. **View results**
   - Total frames processed
   - Detected signs list
   - Confidence scores
   - Time stamps

---

## 🎛️ Features Guide

### Emotion Detection Dropdown

**Click the emotion summary** in top bar:
```
😐 neutral | 65% ▼
```

**See details:**
- Current emotion with emoji
- Confidence percentage
- Valence (happiness scale)
- Arousal (energy scale)
- All emotion probabilities
- Fine-tune button for adjustments

### Camera Controls

**📹 Camera dropdown**: Select camera device
**Switch Camera**: Change camera
**Test Connection**: Check server health
**⚥ Gender**: Toggle gender detection

### Translation Display

**Features:**
- Real-time feed (auto-scrolling)
- Last 50 translations shown
- Each translation shows:
  - Timestamp
  - Sign name with emoji
  - Confidence percentage
- Audio plays automatically

---

## 🔧 Troubleshooting

### No Signs Detected

**Problem**: No translations appearing

**Solutions:**
1. Check camera permission
2. Ensure good lighting
3. Position hand clearly in frame
4. Make gestures more distinct
5. Hold gesture for 2-3 seconds

### Low Confidence

**Problem**: Confidence scores below 50%

**Solutions:**
1. Adjust via API: `POST /signs/confidence/0.5`
2. Make gestures more deliberately
3. Improve lighting
4. Get closer to camera
5. Try different camera angle

### Camera Not Working

**Problem**: Black screen or no video

**Solutions:**
1. Check browser permissions
2. Refresh page and allow camera
3. Try different browser (Chrome recommended)
4. Check if camera is used by another app
5. Select different camera from dropdown

### Server Won't Start

**Problem**: Error when starting server

**Solutions:**
1. Check Python version: `python --version` (need 3.8+)
2. Install dependencies: `pip install -r requirements.txt`
3. Check port availability: `netstat -an | grep 8000`
4. Try different port: `--port 8001`

### Video Upload Fails

**Problem**: Can't upload or process video

**Solutions:**
1. Check file format (MP4 preferred)
2. Reduce file size (< 100MB)
3. Ensure opencv-python installed
4. Check server logs for errors
5. Try shorter video first

---

## 📊 Understanding Results

### Confidence Scores

- **80-100%**: Excellent recognition, very confident
- **60-80%**: Good recognition, reliable
- **40-60%**: Fair recognition, use caution
- **20-40%**: Poor recognition, likely incorrect
- **0-20%**: Very poor, probably wrong

### Translation Quality

**High Quality** (85%+ confidence):
- Clear, distinct gesture
- Good lighting conditions
- Proper hand positioning
- Steady camera

**Medium Quality** (60-85% confidence):
- Slightly unclear gesture
- Okay lighting
- Hand partially in frame
- Minor camera shake

**Low Quality** (< 60% confidence):
- Unclear or wrong gesture
- Poor lighting
- Hand out of frame
- Significant movement

---

## 💡 Tips for Best Results

### Lighting
✅ Front-facing light source
✅ Even lighting on hands
✅ Natural daylight is best
❌ Backlighting (light behind you)
❌ Harsh shadows
❌ Dim/dark conditions

### Hand Position
✅ Center of frame
✅ Clear view of all fingers
✅ Steady, not moving quickly
✅ One hand at a time
❌ Edge of frame
❌ Fingers cut off
❌ Rapid movements
❌ Both hands (current limitation)

### Camera
✅ Good quality webcam
✅ 720p or better
✅ Clean lens
✅ Stable position
❌ Low quality camera
❌ Moving camera
❌ Dirty/blurry lens

### Gestures
✅ Hold for 1-2 seconds
✅ Clear, distinct shape
✅ Practice in mirror first
✅ Consistent positioning
❌ Quick movements
❌ Unclear shapes
❌ Constantly changing

---

## 🎓 Learning ASL Signs

### Resources
1. **ASL Alphabet**: Search "ASL alphabet chart"
2. **Common Words**: Search "ASL common phrases"
3. **Practice Videos**: YouTube "ASL practice"
4. **Online Courses**: Free ASL courses online

### Practice Flow
1. Learn sign from chart/video
2. Practice in front of mirror
3. Try with SignL system
4. Adjust based on confidence
5. Repeat until consistent 80%+

### Current Supported Signs

**Alphabet**: A, B, C, D, E, F, G, H, I

**Words**: 
- hello, thanks, yes, no, please
- sorry, help, good, bad, love

**Gestures**:
- thumbs_up, peace, okay, stop, come

*More signs coming in future updates!*

---

## 📞 Getting Help

### Check Documentation
1. `IMPLEMENTATION_SUMMARY.md` - Technical details
2. `VISUAL_GUIDE.md` - Visual explanations
3. `docs/SIGN_LANGUAGE.md` - Feature documentation
4. `README.md` - General information

### API Documentation
Open in browser:
```
http://localhost:8000/docs
```

Interactive API testing interface with all endpoints.

### Common Issues
Check `IMPLEMENTATION_SUMMARY.md` → Troubleshooting section

### Report Issues
Contact project owner with:
- Description of problem
- Steps to reproduce
- Screenshots if applicable
- Browser and OS info
- Server logs if available

---

## 🎉 Success Criteria

You'll know it's working when:

✅ Camera shows your video feed
✅ Processed feed shows face mesh/landmarks
✅ Making gestures shows translations
✅ Audio plays with each translation
✅ Confidence scores > 70% for clear signs
✅ Translation history scrolls automatically
✅ Emotion dropdown shows current emotion

---

## 🚦 Status Indicators

### Connection Status
- 🟢 **Green**: "Connected - Streaming Active" → All good!
- 🟡 **Yellow**: "Connecting..." → Wait a moment
- 🔴 **Red**: "Disconnected" → Check server

### Sign Detection
- ✅ Translation appears → Sign recognized
- ⏳ "No sign detected" → Keep trying
- ❌ Low confidence → Improve gesture

### Video Stats
- **FPS**: Should be 15-30
- **Faces**: Number detected
- **Sign**: Current prediction

---

## 🎯 Next Steps

Once you're comfortable with basics:

1. **Test all supported signs**
2. **Upload test videos**
3. **Experiment with camera angles**
4. **Try different lighting**
5. **Practice clear gestures**
6. **Adjust confidence threshold**
7. **Provide feedback for improvements**

---

**Have fun exploring sign language translation! 🤟**

For detailed technical information, see `IMPLEMENTATION_SUMMARY.md`
For visual guides, see `VISUAL_GUIDE.md`
For API reference, visit `/docs` endpoint
