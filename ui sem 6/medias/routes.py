import os
import secrets
from PIL import Image
from flask import  render_template, url_for, flash, redirect, request, abort
from medias import app, db, bcrypt
from medias.forms import RegistrationForm, LoginForm, UpdateAccountForm, PostForm ,Comment
from medias.models import User,Post,Comments
from flask_login import login_user, current_user, logout_user, login_required




@app.route("/")
@app.route("/home")
def home():
    posts = Post.query.all()
    return render_template('home.html', posts=posts)

@app.route("/register" , methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created !','success')
        return redirect(url_for('login'))
    return render_template('register.html' , title='Register' , form=form)

@app.route("/login", methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('login unsuccessful' , 'danger')
       

    return render_template('login.html' , title='Login' , form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)
    
    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size) 

    i.save(picture_path)
    return picture_fn

def save_pic(form_pic):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_pic.filename)
    pic_fn = random_hex + f_ext
    pic_path = os.path.join(app.root_path, 'static/post_pics', pic_fn)
    output_size = (500, 500)
    i = Image.open(form_pic)
    i.thumbnail(output_size) 

    i.save(pic_path)
    
    return pic_fn

@app.route("/account", methods=['GET','POST'])
@login_required
def account():
    form = UpdateAccountForm() 
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file

        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('your account has been updated', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)

    posts = Post.query.filter_by(user_id=current_user.id).all()

    return render_template('account.html', title='Account', image_file=image_file, form=form, posts=posts )

@app.route("/post/new", methods=['GET','POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post_pic = save_pic(form.pic.data)
        pic = post_pic
        post = Post(title=form.title.data, content=form.content.data, pic=pic, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('your post is created', 'success')
        pic = url_for('static', filename='profile_pics/' + pic)
        return render_template('home.html', title='Account', pic=pic, form=form )
    return render_template('create.html',form=form, legend='Add post')
@app.route("/user/<int:user_id>",methods=['GET','POST'])
def user(user_id):
    user=User.query.get_or_404(user_id)
    posts = Post.query.filter_by(user_id=user_id).all()
    return render_template('user.html',name=user.username,email=user.email,post=user.posts,img=user.image_file,posts=posts)

@app.route("/post/<int:post_id>/",methods=['GET','POST'])
def post(post_id):
    form = Comment()
    post = Post.query.get_or_404(post_id)
    comments=Comments.query.filter_by(post_id=post_id).all()
    if form.comment.data:
        com= Comments(comments=form.comment.data,post_id=post_id,user_id=current_user.id)
        db.session.add(com)
        db.session.commit()
        flash('Commented !', 'success')
    return render_template('post.html', title=post.title, post=post, comments=comments, form=form)
