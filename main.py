import bcrypt
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Field, Session, SQLModel, create_engine, select
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Optional, Dict
from datetime import datetime, timedelta, date
from jose import JWTError, jwt
from collections import defaultdict
import os

# --- НАСТРОЙКИ ---
SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)

# --- МОДЕЛИ ---
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str = "user"
    full_name: Optional[str] = None
    position: Optional[str] = None
    salary: Optional[int] = None
    start_date: Optional[date] = None
    last_raise_date: Optional[date] = None
    project: Optional[str] = None
    team_lead: Optional[str] = None
    manager: Optional[str] = None
    probation_end_date: Optional[date] = None

class UserUpdate(SQLModel):
    full_name: Optional[str] = None
    position: Optional[str] = None
    salary: Optional[int] = None
    start_date: Optional[date] = None
    last_raise_date: Optional[date] = None
    project: Optional[str] = None
    team_lead: Optional[str] = None
    manager: Optional[str] = None
    probation_end_date: Optional[date] = None

class UserRead(SQLModel):
    id: int
    username: str
    role: str
    full_name: Optional[str] = None
    position: Optional[str] = None
    salary: Optional[int] = None
    start_date: Optional[date] = None
    last_raise_date: Optional[date] = None
    project: Optional[str] = None
    team_lead: Optional[str] = None
    manager: Optional[str] = None
    probation_end_date: Optional[date] = None

class UserPasswordChange(SQLModel):
    old_password: str
    new_password: str

class AdminPasswordReset(SQLModel):
    new_password: str

class Goal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: Optional[str] = None   
    action_plan: str                    
    success_metrics: str                
    review_date: str                    
    reviewer_name: str                  
    status: str = "active"          
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")

class GoalUpdate(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None

class Achievement(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    date_added: date = Field(default_factory=date.today)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")

class AchievementCreateAdmin(SQLModel):
    user_id: int
    text: str

class Comment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    goal_id: int = Field(foreign_key="goal.id")
    author_name: str 
    author_role: str

class CommentCreate(SQLModel):
    text: str

class Survey(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    is_active: bool = True
    created_at: date = Field(default_factory=date.today)

class SurveyResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    survey_id: int = Field(foreign_key="survey.id")
    reviewer_id: int = Field(foreign_key="user.id") 
    target_user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    rating: int 
    positive_feedback: Optional[str] = None 
    growth_areas: Optional[str] = None 

class ProjectFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    survey_id: int = Field(foreign_key="survey.id")
    reviewer_id: int = Field(foreign_key="user.id")
    project_name: str 
    rating: int
    comment: Optional[str] = None

class TeamFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    survey_id: int = Field(foreign_key="survey.id")
    reviewer_id: int = Field(foreign_key="user.id")
    team_name: str 
    rating: int
    comment: Optional[str] = None

# --- НОВАЯ МОДЕЛЬ: ЛИЧНЫЙ ФИДБЕК ---
class PersonalFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    survey_id: int = Field(foreign_key="survey.id")
    user_id: int = Field(foreign_key="user.id") # Кто написал
    q1_team_opinion: Optional[str] = None # Вопрос про мнение команды
    q2_process_ideas: Optional[str] = None # Вопрос про процессы
    q3_personal_1on1: Optional[str] = None # Вопрос про личные проблемы (1-1)
    created_at: date = Field(default_factory=date.today)

# --- МОДЕЛИ ДЛЯ ОТВЕТОВ И СТАТИСТИКИ ---
class SurveySubmissionItem(SQLModel):
    target_user_id: int
    rating: int
    positive_feedback: str
    growth_areas: str

class ProjectSubmissionItem(SQLModel):
    project_name: str
    rating: int
    comment: str

class TeamSubmissionItem(SQLModel):
    team_name: str
    rating: int
    comment: str

class PersonalSubmissionItem(SQLModel):
    q1: Optional[str] = None
    q2: Optional[str] = None
    q3: Optional[str] = None

class SurveySubmission(SQLModel):
    survey_id: int
    responses: List[SurveySubmissionItem]
    project_responses: List[ProjectSubmissionItem]
    team_responses: List[TeamSubmissionItem]
    personal_response: Optional[PersonalSubmissionItem] = None 

class FeedbackStats(SQLModel):
    survey_title: str
    self_rating: Optional[float] = 0
    team_rating: Optional[float] = 0

class ProjectStatsItem(SQLModel):
    project_name: str
    avg_rating: float
    comments: List[str]

class ProjectHistoryItem(SQLModel):
    survey_title: str
    sup_rating: float = 0
    drp_rating: float = 0

class TeamStatsItem(SQLModel):
    team_name: str
    avg_rating: float
    comments: List[str]

class TeamHistoryItem(SQLModel):
    survey_title: str
    rating: float = 0

class PersonalFeedbackRead(SQLModel):
    id: int
    survey_title: str
    q1_team_opinion: Optional[str] = None
    q2_process_ideas: Optional[str] = None
    q3_personal_1on1: Optional[str] = None
    created_at: date

# --- ФУНКЦИИ БЕЗОПАСНОСТИ ---
def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    plain_password_bytes = plain_password.encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_password_bytes, hashed_password_bytes)

def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
    except JWTError: raise HTTPException(status_code=401)
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == username)).first()
        if not user: raise HTTPException(status_code=401)
        return user

# --- ИНИЦИАЛИЗАЦИЯ И SEEDING ---
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def seed_data():
    with Session(engine) as session:
        if session.exec(select(User).where(User.username == "admin")).first(): return 

        print("--- База пуста. Наполняем данными... ---")

        # ПОЛЬЗОВАТЕЛИ
        admin = User(username="admin", password_hash=get_password_hash("admin"), role="admin", full_name="Главный Администратор", position="HR Director", salary=5000, start_date=date(2020, 1, 1), project="Internal Ops")
        session.add(admin)
        ivan = User(username="ivan", password_hash=get_password_hash("123"), role="user", full_name="Иван Иванов", position="Senior Frontend Dev", salary=3500, start_date=date(2022, 5, 10), last_raise_date=date(2023, 11, 1), project="TeamGrowth Web", team_lead="Анна Петрова", manager="Сергей Галашов", probation_end_date=date(2022, 8, 10))
        session.add(ivan)
        anna = User(username="anna", password_hash=get_password_hash("123"), role="user", full_name="Анна Петрова", position="Tech Lead / Backend", salary=4200, start_date=date(2021, 3, 15), last_raise_date=date(2023, 6, 1), project="TeamGrowth API", team_lead="Самостоятельная", manager="Сергей Галашов", probation_end_date=date(2021, 6, 15))
        session.add(anna)
        petr = User(username="petr", password_hash=get_password_hash("123"), role="user", full_name="Петр Сидоров", position="QA Automation", salary=2800, start_date=date(2023, 9, 1), project="TeamGrowth Testing", team_lead="Анна Петрова", manager="Сергей Галашов", probation_end_date=date(2023, 12, 1))
        session.add(petr)
        session.commit()
        
        session.refresh(ivan); session.refresh(anna); session.refresh(petr); session.refresh(admin)

        # ЦЕЛИ
        g1 = Goal(title="Миграция на React 18", description="Перевести проект на новую версию.", action_plan="Обновить зависимости.", success_metrics="Сборка без ошибок.", review_date=str(date(2024, 6, 1)), reviewer_name="Анна Петрова", status="active", user_id=ivan.id)
        session.add(g1)
        a1 = Achievement(text="Ускорил загрузку главной страницы на 30%", user_id=ivan.id)
        session.add(a1)

        # ОПРОСЫ
        s1 = Survey(title="Review Jan 24", is_active=False, created_at=date(2024, 1, 15))
        s2 = Survey(title="Review Feb 24", is_active=False, created_at=date(2024, 2, 15))
        s3 = Survey(title="Review Mar 24", is_active=True, created_at=date(2024, 3, 15)) 
        session.add(s1); session.add(s2); session.add(s3)
        session.commit()
        session.refresh(s1); session.refresh(s2)

        # ОТВЕТЫ ПО ЛЮДЯМ
        session.add(SurveyResponse(survey_id=s1.id, reviewer_id=ivan.id, target_user_id=ivan.id, rating=4, positive_feedback="Норм", growth_areas="Нет"))
        session.add(SurveyResponse(survey_id=s1.id, reviewer_id=anna.id, target_user_id=ivan.id, rating=5, positive_feedback="Супер", growth_areas="Нет"))
        
        # ОТВЕТЫ ПО ПРОЕКТАМ
        session.add(ProjectFeedback(survey_id=s1.id, reviewer_id=ivan.id, project_name="SUP", rating=5, comment="Ок"))
        session.add(ProjectFeedback(survey_id=s1.id, reviewer_id=ivan.id, project_name="DRP", rating=3, comment="Сложно"))
        session.add(ProjectFeedback(survey_id=s2.id, reviewer_id=ivan.id, project_name="SUP", rating=3, comment="Баги"))
        session.add(ProjectFeedback(survey_id=s2.id, reviewer_id=ivan.id, project_name="DRP", rating=5, comment="Релиз"))

        # ОТВЕТЫ ПО КОМАНДЕ (Steve) - ИСТОРИЯ
        session.add(TeamFeedback(survey_id=s1.id, reviewer_id=ivan.id, team_name="Steve", rating=5, comment="Отличный старт года"))
        session.add(TeamFeedback(survey_id=s1.id, reviewer_id=anna.id, team_name="Steve", rating=4, comment="Хорошо работаем"))
        session.add(TeamFeedback(survey_id=s2.id, reviewer_id=ivan.id, team_name="Steve", rating=3, comment="Устали"))
        session.add(TeamFeedback(survey_id=s2.id, reviewer_id=anna.id, team_name="Steve", rating=4, comment="Стабильно"))

        # ПРИМЕР ЛИЧНОГО ЗАПРОСА
        session.add(PersonalFeedback(survey_id=s3.id, user_id=ivan.id, q1_team_opinion="Хочу обсудить переход на TypeScript", q2_process_ideas="Меньше митингов", q3_personal_1on1="Нужен совет по карьере"))

        session.commit()
        print("--- Данные успешно загружены! ---")

# --- APP ---
app = FastAPI()
# Разрешаем CORS для ngrok и локальных запусков
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    seed_data()

# --- ЭНДПОИНТЫ API ---
@app.post("/register")
def register_user(user_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        if session.exec(select(User).where(User.username == user_data.username)).first(): raise HTTPException(status_code=400, detail="Пользователь занят")
        role = "admin" if user_data.username == "admin" else "user"
        new_user = User(username=user_data.username, password_hash=get_password_hash(user_data.password), role=role)
        session.add(new_user); session.commit()
        return {"message": "OK"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == form_data.username)).first()
        if not user or not verify_password(form_data.password, user.password_hash): raise HTTPException(status_code=400, detail="Ошибка")
        token = create_access_token(data={"sub": user.username})
        return {"access_token": token, "token_type": "bearer", "role": user.role, "username": user.username}

@app.post("/me/password")
def change_my_password(data: UserPasswordChange, current_user: User = Depends(get_current_user)):
    """Пользователь меняет свой пароль"""
    if not verify_password(data.old_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Неверный старый пароль")
    
    with Session(engine) as session:
        db_user = session.get(User, current_user.id)
        db_user.password_hash = get_password_hash(data.new_password)
        session.add(db_user)
        session.commit()
        return {"ok": True}

@app.get("/users", response_model=List[UserRead])
def get_all_users(current_user: User = Depends(get_current_user)):
    with Session(engine) as session: return session.exec(select(User)).all()

@app.get("/me", response_model=UserRead)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.patch("/admin/users/{user_id}")
def update_user_profile(user_id: int, user_update: UserUpdate, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session:
        db_user = session.get(User, user_id)
        if not db_user: raise HTTPException(status_code=404)
        user_data = user_update.dict(exclude_unset=True)
        for key, value in user_data.items(): setattr(db_user, key, value)
        session.add(db_user); session.commit()
        return {"ok": True}

@app.post("/admin/users/{user_id}/reset-password")
def reset_user_password(user_id: int, data: AdminPasswordReset, current_user: User = Depends(get_current_user)):
    """Админ сбрасывает пароль пользователю"""
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session:
        db_user = session.get(User, user_id)
        if not db_user: raise HTTPException(status_code=404)
        
        db_user.password_hash = get_password_hash(data.new_password)
        session.add(db_user)
        session.commit()
        return {"ok": True}

@app.post("/admin/goals")
def assign_goal(goal: Goal, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session:
        session.add(goal); session.commit(); session.refresh(goal)
        return goal

@app.delete("/goals/{goal_id}")
def delete_goal(goal_id: int, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        session.delete(goal); session.commit()
        return {"ok": True}

@app.get("/admin/users/{user_id}/goals", response_model=List[Goal])
def get_user_goals_admin(user_id: int, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session: return session.exec(select(Goal).where(Goal.user_id == user_id)).all()

@app.get("/goals", response_model=List[Goal])
def get_my_goals(current_user: User = Depends(get_current_user)):
    with Session(engine) as session: return session.exec(select(Goal).where(Goal.user_id == current_user.id)).all()

@app.patch("/goals/{goal_id}")
def update_goal(goal_id: int, goal_update: GoalUpdate, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        db_goal = session.get(Goal, goal_id)
        if goal_update.status: db_goal.status = goal_update.status
        if goal_update.title: db_goal.title = goal_update.title
        if goal_update.description: db_goal.description = goal_update.description
        session.add(db_goal); session.commit(); session.refresh(db_goal)
        return db_goal

@app.get("/achievements", response_model=List[Achievement])
def get_achievements(current_user: User = Depends(get_current_user)):
    with Session(engine) as session: return session.exec(select(Achievement).where(Achievement.user_id == current_user.id)).all()

@app.post("/achievements")
def add_achievement(achievement: Achievement, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        achievement.user_id = current_user.id
        session.add(achievement); session.commit(); session.refresh(achievement)
        return achievement

@app.get("/admin/users/{user_id}/achievements", response_model=List[Achievement])
def get_user_achievements_admin(user_id: int, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session: return session.exec(select(Achievement).where(Achievement.user_id == user_id)).all()

@app.post("/admin/achievements")
def add_achievement_admin(data: AchievementCreateAdmin, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session:
        new_achievement = Achievement(text=data.text, user_id=data.user_id)
        session.add(new_achievement); session.commit(); session.refresh(new_achievement)
        return new_achievement

@app.get("/goals/{goal_id}/comments", response_model=List[Comment])
def get_comments(goal_id: int, current_user: User = Depends(get_current_user)):
    with Session(engine) as session: return session.exec(select(Comment).where(Comment.goal_id == goal_id)).all()

@app.post("/goals/{goal_id}/comments")
def add_comment(goal_id: int, comment: CommentCreate, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        author = current_user.full_name if current_user.full_name else current_user.username
        new_comment = Comment(text=comment.text, goal_id=goal_id, author_name=author, author_role=current_user.role)
        session.add(new_comment); session.commit(); session.refresh(new_comment)
        return new_comment

@app.post("/admin/surveys")
def create_survey(title: str, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session:
        active_surveys = session.exec(select(Survey).where(Survey.is_active == True)).all()
        for s in active_surveys: s.is_active = False; session.add(s)
        new_survey = Survey(title=title)
        session.add(new_survey); session.commit(); session.refresh(new_survey)
        return new_survey

@app.get("/surveys/active", response_model=Optional[Survey])
def get_active_survey(current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        survey = session.exec(select(Survey).where(Survey.is_active == True)).first()
        if survey:
            existing = session.exec(select(SurveyResponse).where(SurveyResponse.survey_id == survey.id, SurveyResponse.reviewer_id == current_user.id)).first()
            if existing: return None
        return survey

@app.post("/surveys/submit")
def submit_survey(submission: SurveySubmission, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        survey = session.get(Survey, submission.survey_id)
        if not survey or not survey.is_active: raise HTTPException(status_code=400, detail="Опрос не активен")
        
        # 1. Люди
        for item in submission.responses:
            session.add(SurveyResponse(survey_id=submission.survey_id, reviewer_id=current_user.id, target_user_id=item.target_user_id, rating=item.rating, positive_feedback=item.positive_feedback, growth_areas=item.growth_areas))
        
        # 2. Проекты
        for p in submission.project_responses:
            session.add(ProjectFeedback(survey_id=submission.survey_id, reviewer_id=current_user.id, project_name=p.project_name, rating=p.rating, comment=p.comment))

        # 3. Команда
        for t in submission.team_responses:
            session.add(TeamFeedback(survey_id=submission.survey_id, reviewer_id=current_user.id, team_name=t.team_name, rating=t.rating, comment=t.comment))

        # 4. Личные вопросы (NEW)
        if submission.personal_response:
            pr = submission.personal_response
            # Сохраняем, только если хотя бы одно поле заполнено
            if pr.q1 or pr.q2 or pr.q3:
                session.add(PersonalFeedback(
                    survey_id=submission.survey_id,
                    user_id=current_user.id,
                    q1_team_opinion=pr.q1,
                    q2_process_ideas=pr.q2,
                    q3_personal_1on1=pr.q3
                ))

        session.commit()
        return {"ok": True}

@app.get("/admin/users/{target_id}/feedback", response_model=List[SurveyResponse])
def get_user_feedback_admin(target_id: int, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session: return session.exec(select(SurveyResponse).where(SurveyResponse.target_user_id == target_id)).all()

@app.get("/me/feedback", response_model=List[SurveyResponse])
def get_my_feedback(current_user: User = Depends(get_current_user)):
    with Session(engine) as session: return session.exec(select(SurveyResponse).where(SurveyResponse.target_user_id == current_user.id)).all()

@app.get("/users/{target_id}/stats", response_model=List[FeedbackStats])
def get_user_stats(target_id: int, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin" and current_user.id != target_id: raise HTTPException(status_code=403)
    with Session(engine) as session:
        responses = session.exec(select(SurveyResponse).where(SurveyResponse.target_user_id == target_id)).all()
        grouped = defaultdict(list)
        for r in responses: grouped[r.survey_id].append(r)
        stats = []
        for survey_id, resps in grouped.items():
            survey = session.get(Survey, survey_id)
            if not survey: continue
            self_r = next((r.rating for r in resps if r.reviewer_id == target_id), 0)
            team_rs = [r.rating for r in resps if r.reviewer_id != target_id]
            team_avg = sum(team_rs) / len(team_rs) if team_rs else 0
            stats.append(FeedbackStats(survey_title=survey.title, self_rating=self_r, team_rating=round(team_avg, 2)))
        return stats

@app.get("/stats/projects", response_model=List[ProjectStatsItem])
def get_project_stats(current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        projects = ["SUP", "DRP"]
        stats = []
        for p in projects:
            feedbacks = session.exec(select(ProjectFeedback).where(ProjectFeedback.project_name == p)).all()
            if feedbacks:
                avg = sum(f.rating for f in feedbacks) / len(feedbacks)
                comments = []
                for f in feedbacks:
                    if f.comment:
                        survey = session.get(Survey, f.survey_id)
                        title = survey.title if survey else "Unknown"
                        comments.append(f"{title}: {f.comment}")
                stats.append(ProjectStatsItem(project_name=p, avg_rating=round(avg, 2), comments=comments))
            else:
                stats.append(ProjectStatsItem(project_name=p, avg_rating=0, comments=[]))
        return stats

@app.get("/stats/projects/history", response_model=List[ProjectHistoryItem])
def get_project_history_stats(current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        surveys = session.exec(select(Survey).order_by(Survey.created_at)).all()
        history = []
        for survey in surveys:
            sup_feedbacks = session.exec(select(ProjectFeedback).where(ProjectFeedback.survey_id == survey.id, ProjectFeedback.project_name == "SUP")).all()
            sup_avg = sum(f.rating for f in sup_feedbacks) / len(sup_feedbacks) if sup_feedbacks else 0
            drp_feedbacks = session.exec(select(ProjectFeedback).where(ProjectFeedback.survey_id == survey.id, ProjectFeedback.project_name == "DRP")).all()
            drp_avg = sum(f.rating for f in drp_feedbacks) / len(drp_feedbacks) if drp_feedbacks else 0
            if sup_avg > 0 or drp_avg > 0:
                history.append(ProjectHistoryItem(survey_title=survey.title, sup_rating=round(sup_avg, 2), drp_rating=round(drp_avg, 2)))
        return history

@app.get("/stats/team", response_model=List[TeamStatsItem])
def get_team_stats(current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        # Пока хардкод имени команды, в будущем можно брать из профиля
        teams = ["Steve"]
        stats = []
        for t in teams:
            feedbacks = session.exec(select(TeamFeedback).where(TeamFeedback.team_name == t)).all()
            if feedbacks:
                avg = sum(f.rating for f in feedbacks) / len(feedbacks)
                comments = []
                for f in feedbacks:
                    if f.comment:
                        survey = session.get(Survey, f.survey_id)
                        title = survey.title if survey else "Unknown"
                        comments.append(f"{title}: {f.comment}")
                stats.append(TeamStatsItem(team_name=t, avg_rating=round(avg, 2), comments=comments))
            else:
                stats.append(TeamStatsItem(team_name=t, avg_rating=0, comments=[]))
        return stats

@app.get("/stats/team/history", response_model=List[TeamHistoryItem])
def get_team_history_stats(current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        surveys = session.exec(select(Survey).order_by(Survey.created_at)).all()
        history = []
        for survey in surveys:
            # Считаем среднее по команде Steve
            feedbacks = session.exec(select(TeamFeedback).where(TeamFeedback.survey_id == survey.id, TeamFeedback.team_name == "Steve")).all()
            if feedbacks:
                avg = sum(f.rating for f in feedbacks) / len(feedbacks)
                history.append(TeamHistoryItem(survey_title=survey.title, rating=round(avg, 2)))
        return history

@app.get("/admin/users/{user_id}/personal_feedback", response_model=List[PersonalFeedbackRead])
def get_user_personal_feedback(user_id: int, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin": raise HTTPException(status_code=403)
    with Session(engine) as session:
        results = session.exec(select(PersonalFeedback, Survey).where(PersonalFeedback.user_id == user_id, PersonalFeedback.survey_id == Survey.id)).all()
        output = []
        for pf, s in results:
            output.append(PersonalFeedbackRead(
                id=pf.id,
                survey_title=s.title,
                q1_team_opinion=pf.q1_team_opinion,
                q2_process_ideas=pf.q2_process_ideas,
                q3_personal_1on1=pf.q3_personal_1on1,
                created_at=pf.created_at
            ))
        return output

# --- РАЗДАЧА СТАТИЧЕСКИХ ФАЙЛОВ (ФРОНТЕНД) ---
# Это должно быть в самом низу, после всех API роутов
if os.path.exists("dist"):
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")
else:
    print("ВНИМАНИЕ: Папка 'dist' не найдена. Фронтенд не будет работать.")