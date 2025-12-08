# ניתוח האפליקציה והמלצות לפיתוח

**תאריך**: 2025-01-XX  
**מצב נוכחי**: אפליקציה פועלת עם תכונות בסיסיות  
**מסמך התייחסות**: ACOUSTIC_ANALYSIS_ROADMAP.md

---

## 1. סיכום המצב הנוכחי

### 1.1 מה שכבר קיים ועובד ✅

#### תשתית בסיסית:
- ✅ **מערכת ויזואליזציה** - GPU-accelerated spectrogram עם VisPy
- ✅ **מערכת הערות (Annotations)** - מלבנים על הספקטרוגרמה
- ✅ **ציור עקומות דופלר** - משתמשים יכולים לצייר פוליליין על הערות
- ✅ **כלי מדידה** - מדידות נקודה לנקודה עם משוב ויזואלי
- ✅ **טבלת הערות** - עם עמודות: ID, File, t_start, t_end, f_min, f_max, **SNR (dB)**, **Slope (Hz/s)**, Label, View, Curve
- ✅ **ניתוח דופלר בסיסי** - curve fitting (עם מגבלות, ראה סעיף 2.7)
- ✅ **FFT spectrum dialog** - צפייה בתדרים של אזורים נבחרים
- ✅ **תפריט הגדרות** - FFT size, overlap, window function
- ✅ **שמירה/טעינה** - JSON per file (`{filename}_annotations.json`)
- ✅ **ייצוא CSV בסיסי** - `export_project_csv()` קיים ב-`AnnotationManager`

#### מבנה נתונים:
- ✅ **Annotation dataclass** - כולל שדות: `snr_db`, `slope_hz_per_sec`, `harmonic_order`, `event_id`
- ✅ **AnnotationManager** - ניהול אוסף הערות
- ✅ **AnnotationTableWidget** - טבלה עם עמודות SNR ו-Slope (מוכן להצגה)

### 1.2 מה שחסר מהרודמאפ ❌

#### Priority 1: Project Management
- ❌ **ProjectManager** - אין ניהול פרויקטים
- ❌ **project.json** - אין מבנה פרויקט מאוחד
- ❌ **file_metadata.json** - אין שמירת פרמטרי ספקטרוגרמה עם הערות
- ❌ **תפריט Project** - אין New/Open/Save Project

#### Priority 2: Track Analysis
- ❌ **TrackAnalyzer** - אין מחלקה לניתוח מסלולים
- ❌ **חישוב SNR** - השדה קיים אבל אין חישוב בפועל
- ❌ **חישוב Slope** - השדה קיים אבל אין חישוב בפועל
- ❌ **Adaptive Bandwidth SNR** - אין יישום
- ❌ **כפתור "Analyze Track"** - אין פעולה ב-UI

#### Priority 3: Harmonic Detection
- ❌ **HarmonicDetector** - אין מחלקה
- ❌ **איתור הרמוניות אוטומטי** - אין
- ❌ **קישור הרמוניות** - השדות קיימים אבל אין לוגיקה

#### Priority 4: Advanced Export & Event Grouping
- ⚠️ **ייצוא CSV בסיסי** - קיים אבל לא מלא
- ❌ **EventGrouper** - אין קיבוץ אירועים
- ❌ **ייצוא תמונות עם צירים** - אין
- ❌ **ייצוא Events CSV** - אין
- ❌ **HTML Report** - אין

#### Priority 5: Automatic Track Detection
- ❌ **AutomaticTrackDetector** - אין
- ❌ **איתור אוטומטי** - אין
- ❌ **Meijering filter integration** - אין

#### Priority 6: GPS Integration
- ❌ **GPS support** - לא מתוכנן לעתיד הקרוב

---

## 2. המלצות לשיפורים

### 2.1 שיפורים מיידיים (Quick Wins)

#### א. הפעלת חישוב SNR ו-Slope
**בעיה**: השדות קיימים ב-`Annotation` וב-`AnnotationTableWidget`, אבל אין חישוב בפועל.

**פתרון**:
1. ליצור `audio_visualizer/core/track_analyzer.py`
2. לממש את הפונקציות מהרודמאפ:
   - `calculate_track_snr()` - עם adaptive bandwidth
   - `analyze_track_slope()` - חישוב שיפוע
3. להוסיף כפתור "Analyze Track" ב-UI
4. לעדכן את הטבלה אוטומטית אחרי ניתוח

**קושי**: בינוני  
**זמן משוער**: 4-6 שעות  
**ערך**: גבוה - זה Priority 2 מהרודמאפ

#### ב. הסרת עמודות לא אמינות מטבלת הערות
**בעיה**: לפי Priority 7, עמודות Speed ו-CPA Distance לא אמינות ללא GPS.

**פתרון**: 
- ✅ כבר נעשה! הטבלה לא כוללת עמודות אלו
- יש לוודא שגם ב-`export_project_csv()` לא נכללות

**קושי**: נמוך  
**זמן משוער**: 30 דקות  
**ערך**: בינוני - שיפור UX

#### ג. שמירת פרמטרי ספקטרוגרמה עם הערות
**בעיה**: אין אפשרות לשחזר תנאי ניתוח מדויקים.

**פתרון**:
1. לעדכן `AnnotationManager.save_to_json()` לשמור גם:
   ```python
   'spectrogram_params': {
       'fft_size': self.spectrogram_engine.fft_size,
       'hop_length': self.spectrogram_engine.hop_length,
       'window': self.spectrogram_engine.window_type
   }
   ```
2. לעדכן `Annotation.analysis_params` בעת שמירה

**קושי**: נמוך  
**זמן משוער**: 1-2 שעות  
**ערך**: גבוה - חשוב למחקר

### 2.2 שיפורים בינוניים (Medium Priority)

#### א. Project Management (Priority 1)
**בעיה**: אין מבנה פרויקט מאוחד, כל קובץ נשמר בנפרד.

**פתרון**:
1. ליצור `audio_visualizer/core/project_manager.py`
2. לממש מבנה תיקיות:
   ```
   project_folder/
   ├── project.json
   ├── files/
   │   └── audio_file_1/
   │       ├── file_metadata.json
   │       └── annotations.json
   └── exports/
   ```
3. להוסיף תפריט Project ב-`main_window.py`

**קושי**: בינוני-גבוה  
**זמן משוער**: 8-12 שעות  
**ערך**: גבוה מאוד - בסיס לכל התכונות הבאות

#### ב. Harmonic Detection (Priority 3)
**בעיה**: אין איתור אוטומטי של הרמוניות.

**פתרון**:
1. ליצור `audio_visualizer/core/harmonic_detector.py`
2. לממש את האלגוריתם מהרודמאפ (סעיף 4.2)
3. להוסיף "Find Harmonics" ב-context menu של הערה

**קושי**: גבוה  
**זמן משוער**: 10-16 שעות  
**ערך**: גבוה - חוסך זמן רב למשתמש

#### ג. Event Grouping (Priority 4)
**בעיה**: אין קיבוץ הערות לאירועים.

**פתרון**:
1. ליצור `audio_visualizer/core/event_grouper.py`
2. לממש קיבוץ לפי:
   - Temporal overlap
   - Harmonic relationships
   - Similar slope patterns
3. להוסיף עמודת Event ID לטבלה

**קושי**: בינוני  
**זמן משוער**: 6-8 שעות  
**ערך**: בינוני-גבוה - שימושי לניתוח

### 2.3 שיפורים מתקדמים (Advanced)

#### א. Automatic Track Detection (Priority 5)
**בעיה**: אין איתור אוטומטי של מסלולי דופלר.

**פתרון**:
1. ליצור `audio_visualizer/core/auto_detector.py`
2. לממש את האלגוריתם המשופר מהרודמאפ (סעיף 6.3):
   - Meijering ridge filter
   - Adaptive thresholding
   - DBSCAN clustering
   - Centerline extraction
3. להוסיף "Auto Detect Tracks" בתפריט

**קושי**: גבוה מאוד  
**זמן משוער**: 16-24 שעות  
**ערך**: גבוה מאוד - חוסך זמן עצום

#### ב. Advanced Export (Priority 4)
**בעיה**: ייצוא בסיסי קיים אבל לא מלא.

**פתרון**:
1. ליצור `audio_visualizer/core/export_manager.py`
2. לממש:
   - `export_annotation_image()` - עם צירים (matplotlib)
   - `export_events_csv()` - ייצוא אירועים
   - `generate_html_report()` - דוח HTML

**קושי**: בינוני  
**זמן משוער**: 8-10 שעות  
**ערך**: בינוני - שימושי למחקר

---

## 3. מה צריך להיות השלב הבא?

### המלצה: להתחיל ב-Priority 2 (Track Analysis)

**למה?**
1. **השדות כבר קיימים** - רק צריך לממש את החישוב
2. **ערך מיידי** - משתמשים יכולים לראות SNR ו-Slope
3. **בסיס לתכונות אחרות** - Harmonic Detection ו-Event Grouping צריכים נתונים אלו
4. **קושי בינוני** - לא מסובך מדי, לא קל מדי

### תוכנית עבודה מפורטת:

#### שלב 1: יצירת TrackAnalyzer (4-6 שעות)
```python
# audio_visualizer/core/track_analyzer.py
class TrackAnalyzer:
    def __init__(self, q_factor: float = 10.0):
        self.q_factor = q_factor
    
    def analyze(self, spectrogram, track_points, times, freqs) -> TrackAnalysisResult:
        """Full track analysis with SNR and slope"""
        snr_result = self.calculate_snr(...)
        slope_result = self.calculate_slope(...)
        return TrackAnalysisResult(snr_db=..., slope_hz_per_sec=...)
    
    def calculate_snr(self, ...) -> float:
        """Adaptive bandwidth SNR calculation"""
        # Implement from roadmap section 3.2.2
    
    def calculate_slope(self, ...) -> dict:
        """Slope analysis"""
        # Implement from roadmap section 3.2.3
```

#### שלב 2: אינטגרציה עם UI (2-3 שעות)
1. להוסיף כפתור "Analyze Track" ב-context menu של הערה
2. לקרוא ל-`TrackAnalyzer.analyze()` עם נתוני הספקטרוגרמה
3. לעדכן את `Annotation.snr_db` ו-`Annotation.slope_hz_per_sec`
4. לעדכן את הטבלה אוטומטית

#### שלב 3: בדיקות (1-2 שעות)
1. לבדוק על קבצי אודיו אמיתיים
2. לוודא שהחישובים הגיוניים
3. לבדוק edge cases (מסלולים קצרים, רעש גבוה)

**סה"כ**: 7-11 שעות עבודה

---

## 4. סדר עדיפויות מומלץ

### Phase 1: Track Analysis (Priority 2) - **התחל כאן!**
- ✅ יצירת `TrackAnalyzer`
- ✅ חישוב SNR ו-Slope
- ✅ אינטגרציה עם UI
- **זמן**: 7-11 שעות
- **ערך**: גבוה מאוד

### Phase 2: Project Management (Priority 1)
- ✅ יצירת `ProjectManager`
- ✅ מבנה פרויקט
- ✅ תפריט Project
- **זמן**: 8-12 שעות
- **ערך**: גבוה (בסיס לכל השאר)

### Phase 3: Harmonic Detection (Priority 3)
- ✅ יצירת `HarmonicDetector`
- ✅ איתור אוטומטי
- ✅ קישור הרמוניות
- **זמן**: 10-16 שעות
- **ערך**: גבוה

### Phase 4: Event Grouping & Export (Priority 4)
- ✅ יצירת `EventGrouper`
- ✅ יצירת `ExportManager`
- ✅ ייצוא תמונות עם צירים
- ✅ ייצוא Events CSV
- **זמן**: 14-18 שעות
- **ערך**: בינוני-גבוה

### Phase 5: Automatic Detection (Priority 5)
- ✅ יצירת `AutomaticTrackDetector`
- ✅ אינטגרציה עם Meijering
- ✅ UI לאיתור אוטומטי
- **זמן**: 16-24 שעות
- **ערך**: גבוה מאוד

---

## 5. שיפורים נוספים מוצעים

### 5.1 שיפורי UX קטנים
1. **Tooltips** - הוספת הסברים לעמודות בטבלה
2. **Keyboard shortcuts** - קיצורי מקלדת לפעולות נפוצות
3. **Progress indicators** - אינדיקטורי התקדמות לניתוחים ארוכים
4. **Undo/Redo** - ביטול פעולות (קשה, אבל שימושי)

### 5.2 שיפורי ביצועים
1. **Caching של ניתוחים** - שמירת תוצאות SNR/Slope
2. **Background processing** - ניתוח ברקע ללא חסימת UI
3. **Batch analysis** - ניתוח כל ההערות בבת אחת

### 5.3 שיפורי איכות קוד
1. **Unit tests** - בדיקות ל-`TrackAnalyzer`, `HarmonicDetector`
2. **Type hints** - הוספת type hints מלאים
3. **Documentation** - תיעוד API מלא

---

## 6. סיכום והמלצות

### מה לעשות עכשיו:
1. ✅ **להתחיל עם Track Analysis (Priority 2)**
   - הכי מהיר להשלמה
   - ערך מיידי גבוה
   - בסיס לתכונות אחרות

2. ✅ **לשפר את שמירת הפרמטרים**
   - שמירת `spectrogram_params` עם כל הערה
   - חשוב לשחזור ניתוחים

3. ✅ **לבדוק את ייצוא CSV הקיים**
   - לוודא שלא כולל עמודות לא אמינות
   - להוסיף עמודות SNR ו-Slope אם חסרות

### מה לדחות:
- ❌ GPS Integration (Priority 6) - לא קריטי כרגע
- ❌ Automatic Detection (Priority 5) - מורכב, אפשר אחרי Track Analysis
- ❌ HTML Reports - נחמד אבל לא קריטי

### סדר העבודה המומלץ:
```
Week 1: Track Analysis (Priority 2)
Week 2: Project Management (Priority 1)
Week 3: Harmonic Detection (Priority 3)
Week 4: Event Grouping & Export (Priority 4)
Week 5+: Automatic Detection (Priority 5)
```

---

## 7. הערות טכניות

### תלויות נדרשות:
- ✅ `numpy`, `scipy` - כבר קיימים
- ✅ `scikit-image` - נדרש ל-Meijering filter (Priority 5)
- ✅ `matplotlib` - נדרש לייצוא תמונות (Priority 4)
- ✅ `pandas` - כבר קיים (ייצוא CSV)

### מבנה קבצים מוצע:
```
audio_visualizer/
├── core/
│   ├── track_analyzer.py          # NEW - Priority 2
│   ├── project_manager.py          # NEW - Priority 1
│   ├── harmonic_detector.py        # NEW - Priority 3
│   ├── event_grouper.py            # NEW - Priority 4
│   ├── auto_detector.py            # NEW - Priority 5
│   └── export_manager.py            # NEW - Priority 4
└── ui/
    └── main_window.py              # UPDATE - להוסיף תפריטים
```

---

**מסמך זה נכתב על בסיס:**
- בדיקת הקוד הקיים
- השוואה ל-ACOUSTIC_ANALYSIS_ROADMAP.md
- הערכת קושי וזמן עבודה
- הערכת ערך למשתמש

**עודכן**: 2025-01-XX

